import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Dashboard SmartMarket",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv('df_final.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Création des métriques calculées
    df['CTR'] = (df['clicks'] / df['impressions']) * 100
    df['conversion_rate'] = (df['conversions'] / df['clicks']) * 100
    df['cost_per_click'] = df['cost'] / df['clicks']
    df['cost_per_conversion'] = df['cost'] / df['conversions']
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['week'] = df['date'].dt.isocalendar().week
    
    return df

df = load_data()

# CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown('<div class="main-header"> Dashboard Marketing - SmartMarket</div>', unsafe_allow_html=True)
st.markdown("### Analyse des Performances des Campagnes Marketing Multi-Canaux")

# Barre latérale - Filtres
st.sidebar.header(" Filtres")

# Filtre par canal
channels = ['Tous'] + list(df['channel'].unique())
selected_channel = st.sidebar.selectbox("Canal Marketing", channels)

# Filtre par device
devices = ['Tous'] + list(df['device'].unique())
selected_device = st.sidebar.selectbox("Type d'Appareil", devices)

# Filtre par secteur
sectors = ['Tous'] + list(df['sector'].unique())
selected_sector = st.sidebar.selectbox("Secteur d'Activité", sectors)

# Filtre par taille d'entreprise
sizes = ['Tous'] + list(df['company_size'].unique())
selected_size = st.sidebar.selectbox("Taille d'Entreprise", sizes)

# Filtre par statut
statuses = ['Tous'] + list(df['status'].unique())
selected_status = st.sidebar.selectbox("Statut du Lead", statuses)

# Application des filtres
df_filtered = df.copy()

if selected_channel != 'Tous':
    df_filtered = df_filtered[df_filtered['channel'] == selected_channel]
if selected_device != 'Tous':
    df_filtered = df_filtered[df_filtered['device'] == selected_device]
if selected_sector != 'Tous':
    df_filtered = df_filtered[df_filtered['sector'] == selected_sector]
if selected_size != 'Tous':
    df_filtered = df_filtered[df_filtered['company_size'] == selected_size]
if selected_status != 'Tous':
    df_filtered = df_filtered[df_filtered['status'] == selected_status]

st.sidebar.markdown("---")
st.sidebar.info(f" **{len(df_filtered):,}** leads filtrés sur {len(df):,} au total")

# KPIs Principaux
st.markdown("## Indicateurs Clés de Performance (KPIs)")

col1, col2, col3, col4, col5, col6 = st.columns(6)

# Calcul des KPIs
total_conversions = df_filtered['conversions'].sum()
global_conv_rate = (df_filtered['conversions'].sum() / df_filtered['clicks'].sum() * 100)

global_ctr = (df_filtered['clicks'].sum() / df_filtered['impressions'].sum() * 100)
total_cost = df_filtered['cost'].sum()
cost_per_conv = total_cost / total_conversions if total_conversions > 0 else 0
conv_per_euro = total_conversions / total_cost if total_cost > 0 else 0

with col1:
    st.metric(" Conversions", f"{int(total_conversions):,}", 
              help="Nombre total de conversions générées")

with col2:
    st.metric("Taux de Conversion", f"{global_conv_rate:.2f}%",
              help="Pourcentage de clics convertis en conversions")

with col3:
    st.metric(" CTR Global", f"{global_ctr:.2f}%",
              help="Click-Through Rate : taux de clic sur les impressions")

with col4:
    st.metric(" Coût Total", f"{total_cost:,.0f}€",
              help="Coût total des campagnes")

with col5:
    st.metric(" Coût/Conversion", f"{cost_per_conv:.2f}€",
              help="Coût moyen pour obtenir une conversion")

with col6:
    st.metric(" Conv./Euro", f"{conv_per_euro:.4f}",
              help="Nombre de conversions par euro dépensé")

st.markdown("---")

# Onglets pour organiser les visualisations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Vue d'Ensemble", 
    " Performance Canaux", 
    " Segmentation Clients",
    " Analyse Temporelle",
    " Analyse Détaillée"
])

# TAB 1: Vue d'Ensemble
with tab1:
    st.markdown("###  Aperçu Général des Performances")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des conversions par canal
        conv_by_channel = df_filtered.groupby('channel')['conversions'].sum().reset_index()
        fig1 = px.pie(conv_by_channel, values='conversions', names='channel',
                     title='Répartition des Conversions par Canal',
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     hole=0.4)
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Distribution des leads par statut (Funnel)
        status_order = ['Lead', 'MQL', 'SQL', 'Client']
        status_counts = df_filtered['status'].value_counts().reindex(status_order, fill_value=0).reset_index()
        status_counts.columns = ['status', 'count']
        
        fig2 = go.Figure(go.Funnel(
            y=status_counts['status'],
            x=status_counts['count'],
            textinfo="value+percent initial",
            marker=dict(color=['#E74C3C', '#F39C12', '#F1C40F', '#27AE60'])
        ))
        fig2.update_layout(title='Funnel de Conversion par Statut',
                          height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Comparaison Mobile vs Desktop
        device_perf = df_filtered.groupby('device').agg({
            'CTR': 'mean',
            'conversion_rate': 'mean'
        }).reset_index()
        
        fig3 = go.Figure(data=[
            go.Bar(name='CTR (%)', x=device_perf['device'], y=device_perf['CTR'],
                   marker_color='steelblue'),
            go.Bar(name='Taux Conversion (%)', x=device_perf['device'], y=device_perf['conversion_rate'],
                   marker_color='coral')
        ])
        fig3.update_layout(title='Performance Mobile vs Desktop',
                          barmode='group',
                          yaxis_title='Pourcentage (%)')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        # Top 5 secteurs par conversions
        top_sectors = df_filtered.groupby('sector')['conversions'].sum().nlargest(5).reset_index()
        
        fig4 = px.bar(top_sectors, x='conversions', y='sector', orientation='h',
                     title='Top 5 Secteurs - Nombre de Conversions',
                     color='conversions',
                     color_continuous_scale='Viridis',
                     text='conversions')
        fig4.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig4.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig4, use_container_width=True)

# TAB 2: Performance Canaux
with tab2:
    st.markdown("### Analyse Détaillée par Canal Marketing")
    
    # Tableau comparatif des canaux
    channel_perf = df_filtered.groupby('channel').agg({
        'cost': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'CTR': 'mean',
        'conversion_rate': 'mean',
        'cost_per_conversion': 'mean'
    }).round(2)
    
    channel_perf.columns = ['Coût Total (€)', 'Impressions', 'Clics', 'Conversions',
                           'CTR Moyen (%)', 'Taux Conv. (%)', 'Coût/Conv. (€)']
    
    st.dataframe(channel_perf.style.background_gradient(cmap='RdYlGn_r', subset=['Coût/Conv. (€)'])
                                   .background_gradient(cmap='RdYlGn', subset=['Taux Conv. (%)', 'CTR Moyen (%)'])
                                   .format({'Coût Total (€)': '{:,.2f}€',
                                           'Impressions': '{:,.0f}',
                                           'Clics': '{:,.0f}',
                                           'Conversions': '{:,.0f}',
                                           'CTR Moyen (%)': '{:.2f}%',
                                           'Taux Conv. (%)': '{:.2f}%',
                                           'Coût/Conv. (€)': '{:.2f}€'}),
                use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CTR par canal
        ctr_channel = df_filtered.groupby('channel')['CTR'].mean().sort_values(ascending=False).reset_index()
        
        fig5 = px.bar(ctr_channel, x='channel', y='CTR',
                     title='Taux de Clic (CTR) par Canal',
                     color='CTR',
                     color_continuous_scale='Blues',
                     text='CTR')
        fig5.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig5.update_layout(showlegend=False, yaxis_title='CTR (%)')
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        # Coût par conversion par canal
        cost_channel = df_filtered.groupby('channel')['cost_per_conversion'].mean().sort_values().reset_index()
        
        fig6 = px.bar(cost_channel, x='channel', y='cost_per_conversion',
                     title='Coût Moyen par Conversion par Canal',
                     color='cost_per_conversion',
                     color_continuous_scale='Reds_r',
                     text='cost_per_conversion')
        fig6.update_traces(texttemplate='%{text:.2f}€', textposition='outside')
        fig6.update_layout(showlegend=False, yaxis_title='Coût (€)')
        st.plotly_chart(fig6, use_container_width=True)
    
    # Matrice croisée Canal x Device
    st.markdown("#### Analyse Croisée : Canal × Device")
    
    pivot_conv = df_filtered.pivot_table(values='conversion_rate', 
                                         index='channel', 
                                         columns='device', 
                                         aggfunc='mean').round(2)
    
    fig7 = px.imshow(pivot_conv,
                    labels=dict(x="Device", y="Canal", color="Taux Conv. (%)"),
                    title='Heatmap : Taux de Conversion par Canal et Device',
                    color_continuous_scale='RdYlGn',
                    text_auto='.2f')
    fig7.update_xaxes(side="top")
    st.plotly_chart(fig7, use_container_width=True)

# TAB 3: Segmentation Clients
with tab3:
    st.markdown("### Segmentation et Performance Client")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance par secteur
        sector_perf = df_filtered.groupby('sector').agg({
            'conversions': 'sum',
            'conversion_rate': 'mean',
            'cost_per_conversion': 'mean'
        }).round(2).reset_index()
        
        fig8 = px.scatter(sector_perf, x='conversion_rate', y='cost_per_conversion',
                         size='conversions', color='sector',
                         title='Performance par Secteur (Taille = Nb Conversions)',
                         labels={'conversion_rate': 'Taux de Conversion (%)',
                                'cost_per_conversion': 'Coût par Conversion (€)'},
                         hover_data=['conversions'])
        fig8.update_layout(height=500)
        st.plotly_chart(fig8, use_container_width=True)
    
    with col2:
        # Performance par taille d'entreprise
        size_order = ['1-10', '10-50', '50-100', '100-500']
        size_perf = df_filtered.groupby('company_size').agg({
            'conversions': 'sum',
            'conversion_rate': 'mean'
        }).reindex(size_order, fill_value=0).reset_index()
        
        fig9 = go.Figure()
        fig9.add_trace(go.Bar(
            x=size_perf['company_size'],
            y=size_perf['conversions'],
            name='Conversions',
            marker_color='lightseagreen',
            yaxis='y',
            offsetgroup=1
        ))
        fig9.add_trace(go.Scatter(
            x=size_perf['company_size'],
            y=size_perf['conversion_rate'],
            name='Taux Conv. (%)',
            marker_color='orange',
            yaxis='y2',
            mode='lines+markers'
        ))
        fig9.update_layout(
            title='Performance par Taille d\'Entreprise',
            yaxis=dict(title='Nombre de Conversions'),
            yaxis2=dict(title='Taux de Conversion (%)', overlaying='y', side='right'),
            hovermode='x unified'
        )
        st.plotly_chart(fig9, use_container_width=True)
    
    # Distribution géographique
    st.markdown("#### Performance par Région")
    
    region_perf = df_filtered.groupby('region').agg({
        'conversions': 'sum',
        'cost': 'sum',
        'conversion_rate': 'mean'
    }).sort_values('conversions', ascending=False).head(10).reset_index()
    
    fig10 = px.bar(region_perf, x='region', y='conversions',
                  title='Top 10 Régions par Nombre de Conversions',
                  color='conversion_rate',
                  color_continuous_scale='Viridis',
                  text='conversions')
    fig10.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig10.update_layout(xaxis_tickangle=-45, yaxis_title='Nombre de Conversions')
    st.plotly_chart(fig10, use_container_width=True)

# TAB 4: Analyse Temporelle
with tab4:
    st.markdown("### Évolution Temporelle des Performances")
    
    # Évolution hebdomadaire
    weekly_data = df_filtered.groupby('week').agg({
        'conversions': 'sum',
        'cost': 'sum',
        'CTR': 'mean',
        'conversion_rate': 'mean'
    }).reset_index()
    
    fig11 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Évolution des Conversions par Semaine',
                       'Évolution du Taux de Conversion par Semaine'),
        vertical_spacing=0.15
    )
    
    fig11.add_trace(
        go.Scatter(x=weekly_data['week'], y=weekly_data['conversions'],
                  mode='lines+markers', name='Conversions',
                  line=dict(color='steelblue', width=3),
                  marker=dict(size=8),
                  fill='tonexty'),
        row=1, col=1
    )
    
    fig11.add_trace(
        go.Scatter(x=weekly_data['week'], y=weekly_data['conversion_rate'],
                  mode='lines+markers', name='Taux Conv. (%)',
                  line=dict(color='coral', width=3),
                  marker=dict(size=8)),
        row=2, col=1
    )
    
    # Ligne de moyenne
    avg_conv_rate = weekly_data['conversion_rate'].mean()
    fig11.add_hline(y=avg_conv_rate, line_dash="dash", line_color="red",
                   annotation_text=f"Moyenne: {avg_conv_rate:.2f}%",
                   row=2, col=1)
    
    fig11.update_xaxes(title_text="Semaine", row=2, col=1)
    fig11.update_yaxes(title_text="Nombre de Conversions", row=1, col=1)
    fig11.update_yaxes(title_text="Taux de Conversion (%)", row=2, col=1)
    fig11.update_layout(height=700, showlegend=True)
    
    st.plotly_chart(fig11, use_container_width=True)
    
    # Comparaison mensuelle des canaux
    monthly_channel = df_filtered.groupby(['month', 'channel'])['conversions'].sum().reset_index()
    
    fig12 = px.line(monthly_channel, x='month', y='conversions', color='channel',
                   title='Évolution des Conversions par Canal',
                   markers=True,
                   labels={'month': 'Mois', 'conversions': 'Nombre de Conversions'})
    fig12.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig12, use_container_width=True)

# TAB 5: Analyse Détaillée
with tab5:
    st.markdown("### Données Détaillées et Exports")
    
    # Sélection des colonnes à afficher
    display_cols = st.multiselect(
        "Sélectionnez les colonnes à afficher",
        df_filtered.columns.tolist(),
        default=['lead_id', 'date', 'channel', 'device', 'cost', 'conversions', 
                'CTR', 'conversion_rate', 'status']
    )
    
    # Affichage du dataframe filtré
    st.dataframe(df_filtered[display_cols].head(100), use_container_width=True)
    
    st.info(f" Affichage de 100 premières lignes sur {len(df_filtered):,} au total")
    
    # Statistiques descriptives
    st.markdown("#### Statistiques Descriptives")
    
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    stats_df = df_filtered[numeric_cols].describe().round(2)
    
    st.dataframe(stats_df, use_container_width=True)
    
    # Export des données
    st.markdown("#### Export des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=" Télécharger les données filtrées (CSV)",
            data=csv,
            file_name='smartmarket_filtered_data.csv',
            mime='text/csv',
        )
    
    with col2:
        # Export du résumé
        summary = df_filtered.groupby('channel').agg({
            'conversions': 'sum',
            'cost': 'sum',
            'CTR': 'mean',
            'conversion_rate': 'mean'
        }).round(2)
        
        summary_csv = summary.to_csv().encode('utf-8')
        st.download_button(
            label=" Télécharger le résumé par canal (CSV)",
            data=summary_csv,
            file_name='smartmarket_channel_summary.csv',
            mime='text/csv',
        )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Dashboard Marketing SmartMarket</strong> | Développé pour l'analyse des campagnes multi-canaux</p>
        <p> Données actualisées en temps réel |  Filtres interactifs disponibles</p>
    </div>
""", unsafe_allow_html=True)
