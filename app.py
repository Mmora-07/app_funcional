import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path

# --- 1. DEFINICIÓN DE LA ARQUITECTURA DEL MODELO ---
# Esta arquitectura ya es la correcta y no necesita cambios.
class RiskNN(nn.Module):
    def __init__(self, num_features, cat_dims, emb_dims, hidden, dropout):
        super().__init__()
        self.emb = nn.ModuleList([
            nn.Embedding(dim, emb) for dim, emb in zip(cat_dims, emb_dims)
        ])
        
        layer_list = []
        current_dim = num_features + sum(emb_dims)
        
        for hidden_size in hidden:
            layer_list.append(nn.Linear(current_dim, hidden_size))
            layer_list.append(nn.BatchNorm1d(hidden_size))
            layer_list.append(nn.GELU())
            layer_list.append(nn.Dropout(dropout))
            current_dim = hidden_size
            
        layer_list.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layer_list)

    def forward(self, x_num, x_cat):
        emb = [m(x_cat[:, i]) for i, m in enumerate(self.emb)]
        x = torch.cat(emb + [x_num], dim=1)
        return self.net(x).squeeze(1)

# --- 2. FUNCIÓN PARA CARGAR EL MODELO (EN CACHÉ) ---
@st.cache_resource
def load_model_and_artifacts():
    """Carga el modelo, metadatos y estadísticas de normalización."""
    base_path = Path(__file__).parent
    model_dir = base_path / "modeloFinal"
    
    with open(model_dir / "model_metadata.json", 'r') as f:
        metadata = json.load(f)
        
    model = RiskNN(
        num_features=metadata['num_features'],
        cat_dims=metadata['cat_dims'],
        emb_dims=metadata['emb_dims'],
        hidden=metadata['hidden_layers'],
        dropout=metadata['dropout'],
    )
    
    model.load_state_dict(torch.load(model_dir / "best_model_final.pth", map_location=torch.device("cpu"), weights_only=True))
    model.eval()
    
    norm_stats = {
        'means': pd.Series({
            'loan_amnt': 14755.26, 'annual_inc': 75035.56, 'dti': 18.14, 'revol_util': 55.0,
            'total_acc': 25.41, 'tot_cur_bal': 139474.49, 'days_since_earliest_cr': 5851.05
        }),
        'stds': pd.Series({
            'loan_amnt': 8435.46, 'annual_inc': 64149.29, 'dti': 8.37, 'revol_util': 24.5,
            'total_acc': 11.84, 'tot_cur_bal': 153749.12, 'days_since_earliest_cr': 2724.08
        })
    }
    
    return model, metadata, norm_stats

# --- 3. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Evaluador de Riesgo Crediticio NN", page_icon="🤖", layout="centered")
st.title("🤖 Evaluador de Riesgo Crediticio (Red Neuronal)")
st.markdown("Ingrese los datos del solicitante para obtener una predicción de riesgo.")
st.markdown("---")

# --- 4. CARGAR MODELO ---
try:
    nn_model, metadata, norm_stats = load_model_and_artifacts()
    st.success("Modelo de Red Neuronal cargado exitosamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# --- 5. FORMULARIO DE ENTRADA (CON OPCIONES COMPLETAS) ---
with st.form("formulario_riesgo_nn"):
    st.subheader("📋 Datos del Solicitante")
    
    col1, col2 = st.columns(2)
    with col1:
        loan_amnt = st.number_input("💰 Monto del Préstamo ($)", min_value=1000.0, value=15000.0, step=500.0)
        annual_inc = st.number_input("📈 Ingreso Anual ($)", min_value=10000.0, value=75000.0, step=1000.0)
        dti = st.number_input("📊 Ratio Deuda/Ingreso (DTI)", min_value=0.0, value=18.0, step=0.1)
        revol_util = st.number_input("📉 Uso de Crédito Revolvente (%)", min_value=0.0, max_value=100.0, value=55.0, step=0.1)
    
    with col2:
        total_acc = st.number_input("📂 Cuentas de Crédito Totales", min_value=0, value=25)
        tot_cur_bal = st.number_input("💼 Saldo Corriente Total ($)", min_value=0.0, value=140000.0, step=1000.0)
        days_since_earliest_cr = st.number_input("⏳ Antigüedad Crediticia (días)", min_value=365, value=5800)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        # CORRECCIÓN: Lista completa de opciones para antigüedad laboral
        emp_length_options = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
        emp_length = st.selectbox("📆 Antigüedad Laboral", options=emp_length_options, index=10)
        
        home_ownership_options = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
        home_ownership = st.selectbox("🏠 Tipo de Vivienda", options=home_ownership_options, index=2)
        
        purpose_options = ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase']
        purpose = st.selectbox("🎯 Propósito del Préstamo", options=purpose_options, index=0)
    
    with col4:
        # CORRECCIÓN: Lista completa de opciones para morosidades
        delinq_2yrs_options = ['0', '1', '>=2'] 
        delinq_2yrs = st.selectbox("❌ Morosidades (últimos 2 años)", options=delinq_2yrs_options, index=0)
        
        inq_last_6mths_options = ['0', '1', '>=2']
        inq_last_6mths = st.selectbox("📌 Solicitudes de Crédito (últimos 6 meses)", options=inq_last_6mths_options, index=1)
        
        pub_rec_options = ['0', '1'] 
        pub_rec = st.selectbox("⚖️ Registros Públicos Negativos", options=pub_rec_options, index=0)

    enviado = st.form_submit_button("Evaluar Riesgo")

# --- 6. PROCESAMIENTO Y PREDICCIÓN ---
if enviado:
    # Mapeos para convertir texto a índices
    cat_map = {
        'emp_length': {val: i for i, val in enumerate(emp_length_options)},
        'home_ownership': {val: i for i, val in enumerate(home_ownership_options)},
        'purpose': {val: i for i, val in enumerate(purpose_options)},
        'delinq_2yrs': {val: i for i, val in enumerate(delinq_2yrs_options)},
        'inq_last_6mths': {val: i for i, val in enumerate(inq_last_6mths_options)},
        'pub_rec': {val: i for i, val in enumerate(pub_rec_options)}
    }
    cat_inputs = [
        cat_map['emp_length'][emp_length], cat_map['home_ownership'][home_ownership],
        cat_map['purpose'][purpose], cat_map['delinq_2yrs'][str(delinq_2yrs)], # Convertir a str para el mapeo
        cat_map['inq_last_6mths'][str(inq_last_6mths)], # Convertir a str para el mapeo
        cat_map['pub_rec'][str(pub_rec)] # Convertir a str para el mapeo
    ]
    tensor_cat = torch.tensor([cat_inputs], dtype=torch.long)

    # Procesamiento de Variables Numéricas
    num_inputs_df = pd.DataFrame([[
        loan_amnt, annual_inc, dti, revol_util, total_acc, tot_cur_bal, days_since_earliest_cr
    ]], columns=list(norm_stats['means'].index))
    
    num_inputs_normalized = (num_inputs_df - norm_stats['means']) / norm_stats['stds']
    num_inputs_final = np.insert(num_inputs_normalized.values, 0, 0, axis=1)
    tensor_num = torch.tensor(num_inputs_final, dtype=torch.float32)

    # Realizar la Predicción
    with torch.no_grad():
        logits = nn_model(tensor_num, tensor_cat)
        probability = torch.sigmoid(logits).item()

    # Mostrar el Resultado
    st.markdown("---")
    prob_percent = probability * 100
    
    if probability > 0.5:
        st.error(f"🔴 Riesgo Alto (Probabilidad de Incumplimiento: {prob_percent:.2f}%)")
        st.warning("Recomendación: Es probable que el crédito sea rechazado.")
    else:
        st.success(f"🟢 Riesgo Bajo (Probabilidad de Incumplimiento: {prob_percent:.2f}%)")
        st.info("Recomendación: Es probable que el crédito sea aprobado.")
        st.balloons()
        
    st.progress(probability)
    
    with st.expander("Ver detalles técnicos"):
        st.write(f"Logit del modelo (salida cruda): {logits.item():.4f}")

