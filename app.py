import dash
from dash import html, dcc
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys

# Inicializa o app
app = dash.Dash(__name__, title="Prophet - O Futuro")
server = app.server # server Gunicorn 

arquivo = "dados.xlsx"

# definir valores padrao para as variaveis
kpi_vals = {
    "acc_model": "0%", "mae": "0", "mape": "0%",
    "ret_prev": "0%", "ret_meta": "0%", "ret_real": "0%",
    "delta": "0%", "acc_atual": "0%"
}

data_fim_treino = datetime.date.today()
data_fim_real = datetime.date.today()

fig_teste = go.Figure()
fig_previsao = go.Figure()

try:
    print("--- INICIANDO LEITURA ---")
    # ler as abas do arquivo
    df_metricas = pd.read_excel(arquivo, sheet_name="Metricas do Modelo", engine='openpyxl')
    df_treino = pd.read_excel(arquivo, sheet_name="Dados de Treino", engine='openpyxl')
    df_teste = pd.read_excel(arquivo, sheet_name="Ultimo Mes", engine='openpyxl')
    df_previsao = pd.read_excel(arquivo, sheet_name="Previsoes do Modelo", engine='openpyxl')
    df_real_atual = pd.read_excel(arquivo, sheet_name="Mes Atual Real", engine='openpyxl')

    # normalizar as datas para o merge
    print("Normalizando datas...")
    df_treino['ds'] = pd.to_datetime(df_treino['ds']).dt.normalize()
    df_teste['ds'] = pd.to_datetime(df_teste['ds']).dt.normalize()
    df_previsao['ds'] = pd.to_datetime(df_previsao['ds']).dt.normalize()
    df_real_atual['ds'] = pd.to_datetime(df_real_atual['ds']).dt.normalize()

    # datas máximas para o cabecalho
    if not df_treino.empty:
        data_fim_treino = df_treino['ds'].max().date()
    
    if not df_real_atual.empty:
        data_fim_real = df_real_atual['ds'].max().date()

    # transformar as metricas em string
    df_metricas['Metrica'] = df_metricas['Metrica'].astype(str).str.strip()
    
    # pegar as metricas na aba de Metricas
    def pegar_metrica(nome):
        row = df_metricas[df_metricas['Metrica'] == nome]
        return row['Valor'].values[0] if not row.empty else 0

    acerto_val = pegar_metrica("Taxa Acerto Intervalo %")
    kpi_vals["acc_model"] = f"{acerto_val * 100:.2f}%" if acerto_val <= 1 else f"{acerto_val:.2f}%"
    
    mae_val = pegar_metrica("MAE")
    kpi_vals["mae"] = f"{mae_val * 100:.2f}%" if mae_val <= 1 else f"{mae_val:.2f}%"
    
    mape_val = pegar_metrica("MAPE")
    kpi_vals["mape"] = f"{mape_val * 100:.2f}%" if mape_val <= 1 else f"{mape_val:.2f}%"

    # fazendo merge da previsao com os dados atuais disponiveis para fazer os cards de retencao prevista, meta, etc
    print("Realizando Merge dos dados...")
    df_monitor = pd.merge(
        df_real_atual, 
        df_previsao[['ds', 'yhat', 'Meta_Diaria', 'yhat_lower', 'yhat_upper']], 
        on='ds', 
        how='inner'
    )
    
    # exibir no cmd pra validar se deu erro
    print(f"Linhas encontradas no cruzamento Real x Previsto: {len(df_monitor)}")

    if not df_monitor.empty:
        total_acumulado = df_monitor['TOTAL'].sum()
        
        if total_acumulado > 0:
            # primeiro a retencao prevista
            vol_previsto = (df_monitor['TOTAL'] * df_monitor['yhat']).sum()
            taxa_prevista_pond = vol_previsto / total_acumulado
            kpi_vals["ret_prev"] = f"{taxa_prevista_pond * 100:.2f}%"

            # segundo o que eles deviam estar fazendo (meta)
            vol_meta = (df_monitor['TOTAL'] * df_monitor['Meta_Diaria']).sum()
            taxa_meta_pond = vol_meta / total_acumulado
            kpi_vals["ret_meta"] = f"{taxa_meta_pond * 100:.2f}%"

            # terceiro retencao real performada
            retido_real_total = df_monitor['RETIDO'].sum()
            taxa_real_consol = retido_real_total / total_acumulado
            kpi_vals["ret_real"] = f"{taxa_real_consol * 100:.2f}%"

            # delta medio entre real e previsto
            df_monitor['taxa_real_dia'] = df_monitor['RETIDO'] / df_monitor['TOTAL']
            df_monitor['delta_dia'] = df_monitor['taxa_real_dia'] - df_monitor['yhat']
            media_delta = df_monitor['delta_dia'].mean()
            kpi_vals["delta"] = f"{media_delta * 100:+.2f}%"

            # acerto atual
            df_monitor['dentro_intervalo'] = (df_monitor['taxa_real_dia'] >= df_monitor['yhat_lower']) & \
                                             (df_monitor['taxa_real_dia'] <= df_monitor['yhat_upper'])
            acc_atual_val = df_monitor['dentro_intervalo'].mean()
            kpi_vals["acc_atual"] = f"{acc_atual_val * 100:.1f}%"
    else:
        print("ALERTA: O Merge retornou vazio! Verifique se as datas do Excel coincidem.")

    # criar os graficos
    def criar_graficos(df, titulo):
        fig = go.Figure()
        
        # se tiver vazio, o grafico fica vazio para nao quebrar
        if df.empty:
            return fig

        # as linhas dos graficos
        if 'y' in df.columns:
            fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='REAL', mode='lines',
                line=dict(color='#FF00FF', width=3, shape='spline', smoothing=1.3)))

        fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat'], name='PREVISTO', mode='lines',
            line=dict(color='#00BFFF', width=3, shape='spline', smoothing=1.3)))

        fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat_lower'], name='ABAIXO', mode='lines',
            line=dict(color='#FF4136', width=2, dash='dot', shape='spline')))

        fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat_upper'], name='ACIMA', mode='lines',
            line=dict(color='#2ECC40', width=2, dash='dot', shape='spline')))

        # calculos dos ranges
        vals = df.select_dtypes(include=['number']).values.flatten()
        vals = vals[~pd.isna(vals)]
        range_y = None
        if len(vals) > 0:
            margem = (vals.max() - vals.min()) * 0.1
            range_y = [vals.min() - margem, vals.max() + margem]

        range_x = None
        if not df['ds'].empty:
            delta = df['ds'].max() - df['ds'].min()
            buff = delta * 0.03 if delta > pd.Timedelta(0) else pd.Timedelta(days=1)
            range_x = [df['ds'].min() - buff, df['ds'].max() + buff]

        fig.update_layout(
            title=dict(text=titulo, font=dict(family="Orbitron", size=20, color="#ffffff"), x=0.5),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0, font=dict(color="#aabce0")),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showline=False, color='#aabce0', tickformat='%d/%m', 
                       range=range_x, automargin=True, showspikes=True, spikecolor="rgba(0, 255, 255, 0.3)", spikemode="across"),
            yaxis=dict(showgrid=False, zeroline=False, showline=False, color='#aabce0', tickformat='.0%', 
                       range=range_y, automargin=True),
            margin=dict(l=60, r=40, t=80, b=60),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="rgba(5, 10, 48, 0.95)", bordercolor="#00FFFF", font_size=14, font_family="Roboto", font_color="#ffffff")
        )
        fig.update_traces(hovertemplate="%{y:.1%}")
        return fig

    print("Gerando gráficos...")
    fig_teste = criar_graficos(df_teste, "Teste - Real vs Previsto (Último Mês)")
    fig_previsao = criar_graficos(df_previsao, "Previsões do Modelo (Futuro)")
    print("Sucesso!")

except Exception as e:
    print("!!! ERRO NA EXECUÇÃO !!!")
    print(e)
    import traceback
    traceback.print_exc()


# layout e dados dos Kpis para os cards
kpi_dados = [

    {"label": "Taxa de Acerto", "value": kpi_vals["acc_model"], "desc": "A Taxa de acerto (acurácia) foi calculada com base no intervalo de confiança do modelo (limites superior e inferior da previsão). Considera-se que o modelo acertou quando o valor real observado está dentro desse intervalo.", "align": "tooltip-align-left"},

    {"label": "Erro Médio (MAE)", "value": kpi_vals["mae"], "desc": "O MAE mede o erro médio absoluto entre os valores previstos e os valores reais. Ou seja, indica o quanto, em média, as previsões se afastam do valor verdadeiro. Quanto menor o MAE, melhor o modelo.", "align": ""},

    {"label": "Erro % (MAPE)", "value": kpi_vals["mape"], "desc": "Erro Percentual Absoluto Médio. Indica o quanto o modelo erra em porcentagem em relação ao valor real.", "align": ""},

    {"label": "Taxa de Acerto (Atual)", "value": kpi_vals["acc_atual"], "desc": "A Taxa de acerto (acurácia) foi calculada com base no intervalo de confiança do modelo (limites superior e inferior da previsão). Considera-se que o modelo acertou quando o valor real observado está dentro desse intervalo.", "align": "tooltip-align-right"},
    
    {"label": "Retenção Prevista", "value": kpi_vals["ret_prev"], "desc": "É a previsão do modelo até o momento, de acordo com os dados disponíveis.", "align": "tooltip-align-left"},

    {"label": "Retenção Meta", "value": kpi_vals["ret_meta"], "desc": "É o valor que deveria estar sendo performado para atingir a meta proposta.", "align": ""},

    {"label": "Retenção Real", "value": kpi_vals["ret_real"], "desc": "Resultado Real Consolidado.", "align": ""},

    {"label": "Delta (Real vs Prev)", "value": kpi_vals["delta"], "desc": "Diferença média do Delta (Real - Previsto).", "align": "tooltip-align-right"},

]

app.layout = html.Div(className='main-container', children=[
    html.Div(className='header-container', children=[
        html.Div(className='logo-area', children=[
            html.Img(src='assets/Hub Logo.png', className='logo-img'),
            html.Img(src='assets/Logo - Editado.png', className='logo-img'),
            html.Img(src='assets/tim-25.png', className='logo-img'),
            html.Img(src='assets/Diretoria Logo.png', className='logo-img'),
        ]),
        html.Div(className='title-area', children=[
            html.H1("PROPHET - PREVISÕES", className='main-title'),
            html.P("Dashboard de Previsão", className='subtitle')
        ]),
        
        # area de data no cabecalho
        html.Div(className='date-area', children=[
            
            html.Label("Fim Treino:", className='date-label'),
            html.Div(
                children=data_fim_treino.strftime('%d/%m/%Y'), 
                className='static-date-box'
            ),
            
            html.Label("Fim Dados Reais:", className='date-label', style={'marginTop': '10px'}),
            html.Div(
                children=data_fim_real.strftime('%d/%m/%Y'),   
                className='static-date-box'
            )
        ])
    ]),
    html.Div(className='kpi-grid', children=[
        html.Div(className='kpi-card', children=[
            html.H3(item['label'], className='kpi-label'),
            html.H2(item['value'], className='kpi-value'),
            html.Div(className=f"tooltip-text {item['align']}", children=item['desc']),
            html.Div(className='glow-effect')
        ]) for item in kpi_dados
    ]),
    html.Div(className='charts-container', children=[
        html.Div(className='graph-card', children=[dcc.Graph(figure=fig_teste, config={'displayModeBar': False})]),
        html.Div(className='graph-card', children=[dcc.Graph(figure=fig_previsao, config={'displayModeBar': False})]),
    ])
])

if __name__ == '__main__':
    app.run(debug=False)