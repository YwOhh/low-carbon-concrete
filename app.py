import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ====================== 核心配置 ======================
np.random.seed(42)
FEATURE_COLUMNS_ORIGINAL = [
    'OPC (kg/m3)', 'S (kg/m3)', 'W/B', 'FA (kg/m3)', 'GS (kg/m3)', 'SF (kg/m3)',
    'SP (kg/m3)', 'HPMC (kg/m3)', 'W (kg/m3)', 'Fvol (%)f', 'CD（d)', 'LD (X,Y,Z)',
    'Strength （GPa）', 'Elastic Modulus (GPa)', 'Density (g/cm3)', 'Lf/Df', 'Df (μm)', 'Lf (mm)'
]
TARGET_COLUMN_ORIGINAL = 'CS (MPa)'
CARBON_THRESHOLD = 600  # 低碳阈值：kg CO2 eq/m³
REFERENCE_EMISSION = 662.8490  # 基准碳排放量

# ====================== 碳排放计算函数 ======================
def calculate_emission(row):
    """计算单条配合比的碳排放量"""
    # 各组分GWP系数（kg CO2 eq/kg）
    GWP = {
        'OPC (kg/m3)': 0.925754987,
        'S (kg/m3)': 0.096949054,
        'FA (kg/m3)': 0.035101155,
        'SF (kg/m3)': 0.306808295,
        'GS (kg/m3)': 0.004197845,
        'W (kg/m3)': 0.000552102,
        'Fvol (%)f': 0.027134144,
        'ADD': 0.940857761  # SP+HPMC合并
    }
    
    e_opc = row['OPC (kg/m3)'] * GWP['OPC (kg/m3)'] if 'OPC (kg/m3)' in row else 0
    e_s = row['S (kg/m3)'] * GWP['S (kg/m3)'] if 'S (kg/m3)' in row else 0
    e_fa = row['FA (kg/m3)'] * GWP['FA (kg/m3)'] if 'FA (kg/m3)' in row else 0
    e_sf = row['SF (kg/m3)'] * GWP['SF (kg/m3)'] if 'SF (kg/m3)' in row else 0
    e_gs = row['GS (kg/m3)'] * GWP['GS (kg/m3)'] if 'GS (kg/m3)' in row else 0
    e_add = (row['SP (kg/m3)'] + row['HPMC (kg/m3)']) * GWP['ADD'] if 'SP (kg/m3)' in row and 'HPMC (kg/m3)' in row else 0
    e_fiber = row['Fvol (%)f'] * GWP['Fvol (%)f'] if 'Fvol (%)f' in row else 0
    e_water = row['W (kg/m3)'] * GWP['W (kg/m3)'] if 'W (kg/m3)' in row else 0
    
    return e_opc + e_s + e_fa + e_sf + e_gs + e_add + e_fiber + e_water

# ====================== 数据加载与预处理 ======================
@st.cache_resource
def load_and_preprocess_data():
    """加载数据并完成预处理（缓存避免重复计算）"""
    # 读取数据（Streamlit部署时需确保data.xlsx在同一目录）
    df = pd.read_excel('data.xlsx')
    st.success(f"数据加载成功！共 {df.shape[0]} 条记录，{df.shape[1]} 个字段")
    
    # 列名映射（兼容原始数据格式）
    column_mapping = {col: col for col in FEATURE_COLUMNS_ORIGINAL + [TARGET_COLUMN_ORIGINAL]}
    for expected_col, actual_col in column_mapping.items():
        if actual_col not in df.columns:
            for df_col in df.columns:
                if expected_col.lower().replace(' ', '').replace('（', '(').replace('）', ')') in 
                   df_col.lower().replace(' ', '').replace('（', '(').replace('）', ')'):
                    column_mapping[expected_col] = df_col
                    break
    
    # 提取特征和目标列
    feature_columns = [column_mapping[col] for col in FEATURE_COLUMNS_ORIGINAL]
    target_column = column_mapping[TARGET_COLUMN_ORIGINAL]
    
    # 计算特征统计信息
    feature_stats = {}
    for col in feature_columns:
        if col in df.columns:
            col_data = df[col]
            non_zero_data = col_data[col_data > 0]
            non_zero_min = non_zero_data.min() if len(non_zero_data) > 0 else 0
            
            q1 = col_data.quantile(0.05)
            q3 = col_data.quantile(0.95)
            iqr = q3 - q1
            min_val = max(0, q1 - 1.5 * iqr)
            max_val = q3 + 1.5 * iqr
            
            # 特殊参数约束
            if 'Fvol' in col or 'fvol' in col.lower():
                min_val = max(min_val, non_zero_min * 0.5 if non_zero_min > 0 else 0.1)
                max_val = min(max_val, 10)
            elif 'FA' in col and 'FA (kg/m3)' in col:
                min_val = max(min_val, non_zero_min * 0.5 if non_zero_min > 0 else 20)
                max_val = min(max_val, 600)
            elif 'OPC' in col:
                min_val = max(min_val, non_zero_min * 0.5 if non_zero_min > 0 else 100)
                max_val = min(max_val, 800)
            elif 'W' in col and 'W (kg/m3)' in col:
                min_val = max(min_val, non_zero_min * 0.5 if non_zero_min > 0 else 100)
            elif 'SP' in col and 'SP (kg/m3)' in col:
                min_val = max(min_val, non_zero_min * 0.5 if non_zero_min > 0 else 0.1)
            
            feature_stats[col] = {
                'min': float(min_val), 'max': float(max_val), 'non_zero_min': float(non_zero_min),
                'mean': float(col_data.mean()), 'std': float(col_data.std())
            }
    
    # 数据分割与标准化
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_point = len(df_shuffled) // 2
    inn_data = df_shuffled.iloc[:split_point]
    ann_data = df_shuffled.iloc[split_point:]
    
    # INN数据（强度→配合比）
    inn_X = inn_data[[target_column]].values
    inn_y = inn_data[feature_columns].values
    inn_X_scaler = StandardScaler()
    inn_y_scaler = StandardScaler()
    inn_X_scaled = inn_X_scaler.fit_transform(inn_X)
    inn_y_scaled = inn_y_scaler.fit_transform(inn_y)
    
    # ANN数据（配合比→强度）
    ann_X = ann_data[feature_columns].values
    ann_y = ann_data[[target_column]].values
    ann_X_scaler = StandardScaler()
    ann_y_scaler = StandardScaler()
    ann_X_scaled = ann_X_scaler.fit_transform(ann_X)
    ann_y_scaled = ann_y_scaler.fit_transform(ann_y)
    
    return {
        'df': df, 'feature_columns': feature_columns, 'target_column': target_column,
        'feature_stats': feature_stats,
        'inn_X_scaler': inn_X_scaler, 'inn_y_scaler': inn_y_scaler,
        'ann_X_scaler': ann_X_scaler, 'ann_y_scaler': ann_y_scaler,
        'inn_X_scaled': inn_X_scaled, 'inn_y_scaled': inn_y_scaled,
        'ann_X_scaled': ann_X_scaled, 'ann_y_scaled': ann_y_scaled
    }

# ====================== 模型训练 ======================
@st.cache_resource
def train_models(preprocessed_data):
    """训练INN和ANN模型（缓存避免重复训练）"""
    # 提取预处理数据
    inn_X_scaled = preprocessed_data['inn_X_scaled']
    inn_y_scaled = preprocessed_data['inn_y_scaled']
    ann_X_scaled = preprocessed_data['ann_X_scaled']
    ann_y_scaled = preprocessed_data['ann_y_scaled']
    
    # 划分训练集/测试集
    inn_X_train, inn_X_test, inn_y_train, inn_y_test = train_test_split(
        inn_X_scaled, inn_y_scaled, test_size=0.2, random_state=42
    )
    ann_X_train, ann_X_test, ann_y_train, ann_y_test = train_test_split(
        ann_X_scaled, ann_y_scaled, test_size=0.2, random_state=42
    )
    
    # 训练INN模型（随机森林回归）
    inn_model = RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    inn_model.fit(inn_X_train, inn_y_train)
    
    # 训练ANN模型（多层感知机）
    ann_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
        max_iter=2000, random_state=42, learning_rate_init=0.001,
        alpha=0.0001, early_stopping=True, validation_fraction=0.1, verbose=False
    )
    ann_model.fit(ann_X_train, ann_y_train)
    
    # 模型评估
    inn_test_r2 = r2_score(inn_y_test, inn_model.predict(inn_X_test))
    ann_test_r2 = r2_score(ann_y_test, ann_model.predict(ann_X_test))
    st.success(f"模型训练完成！INN测试集R²: {inn_test_r2:.4f}, ANN测试集R²: {ann_test_r2:.4f}")
    
    return {
        'inn_model': inn_model, 'ann_model': ann_model,
        'inn_test_r2': inn_test_r2, 'ann_test_r2': ann_test_r2
    }

# ====================== 约束函数 ======================
def enforce_constraints(mixes_original, feature_columns, feature_stats, cd_value=28):
    """应用配合比参数约束"""
    mixes_original = np.maximum(mixes_original, 0)
    
    # 范围约束
    for i, col in enumerate(feature_columns):
        if col in feature_stats:
            mixes_original[:, i] = np.clip(mixes_original[:, i], feature_stats[col]['min'], feature_stats[col]['max'])
            
            # 关键参数非零约束
            if ('Fvol' in col or 'fvol' in col.lower()) and feature_stats[col]['non_zero_min'] > 0:
                for j in range(len(mixes_original)):
                    if mixes_original[j, i] < feature_stats[col]['non_zero_min'] * 0.1:
                        mixes_original[j, i] = np.random.uniform(
                            feature_stats[col]['non_zero_min'] * 0.5, feature_stats[col]['non_zero_min'] * 2
                        )
            elif ('FA' in col and 'FA (kg/m3)' in col) and feature_stats[col]['non_zero_min'] > 0:
                for j in range(len(mixes_original)):
                    if mixes_original[j, i] < feature_stats[col]['non_zero_min'] * 0.1:
                        mixes_original[j, i] = np.random.uniform(
                            feature_stats[col]['non_zero_min'] * 0.5, feature_stats[col]['non_zero_min'] * 3
                        )
            elif ('OPC' in col) and feature_stats[col]['non_zero_min'] > 0:
                for j in range(len(mixes_original)):
                    if mixes_original[j, i] < feature_stats[col]['non_zero_min'] * 0.1:
                        mixes_original[j, i] = np.random.uniform(
                            feature_stats[col]['non_zero_min'] * 0.5, feature_stats[col]['non_zero_min'] * 2
                        )
    
    # CD值固定为28天
    cd_index = FEATURE_COLUMNS_ORIGINAL.index('CD（d)')
    mixes_original[:, cd_index] = cd_value
    
    # 水胶比约束（0.2-0.8）
    wb_index = FEATURE_COLUMNS_ORIGINAL.index('W/B')
    mixes_original[:, wb_index] = np.clip(mixes_original[:, wb_index], 0.2, 0.8)
    
    # 小数位数处理
    for i, col in enumerate(feature_columns):
        if 'Fvol' in col or 'fvol' in col.lower():
            mixes_original[:, i] = np.round(mixes_original[:, i], 4)
        else:
            mixes_original[:, i] = np.round(mixes_original[:, i], 2)
    
    return np.maximum(mixes_original, 0)

# ====================== 配合比生成函数 ======================
def generate_mixes(target_strength, num_mixes, preprocessed_data, models):
    """逆向生成配合比（不加碳排放约束）"""
    # 提取数据和模型
    inn_model = models['inn_model']
    ann_model = models['ann_model']
    inn_X_scaler = preprocessed_data['inn_X_scaler']
    inn_y_scaler = preprocessed_data['inn_y_scaler']
    ann_X_scaler = preprocessed_data['ann_X_scaler']
    ann_y_scaler = preprocessed_data['ann_y_scaler']
    feature_columns = preprocessed_data['feature_columns']
    feature_stats = preprocessed_data['feature_stats']
    
    # 目标强度标准化
    target_scaled = inn_X_scaler.transform([[target_strength]])
    candidate_mixes = []
    candidate_errors = []
    candidate_percent_errors = []
    
    # 生成候选配合比
    for i in range(num_mixes * 5):  # 生成5倍候选以确保筛选效果
        # 添加噪声增加多样性
        noise_level = 0.05 * (1 + np.random.rand()) if i < num_mixes * 3 else 0.15 * (1 + np.random.rand())
        noise = np.random.normal(0, noise_level, size=target_scaled.shape)
        noisy_target = target_scaled + noise
        
        # 逆向生成配合比
        mix_scaled = inn_model.predict(noisy_target)
        mix_original = inn_y_scaler.inverse_transform(mix_scaled.reshape(1, -1))
        mix_constrained = enforce_constraints(mix_original, feature_columns, feature_stats)
        
        # 正向预测强度验证
        mix_scaled_constrained = ann_X_scaler.transform(mix_constrained)
        pred_scaled = ann_model.predict(mix_scaled_constrained)
        pred = ann_y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        
        # 计算误差
        error = abs(pred - target_strength)
        percent_error = (error / target_strength) * 100 if target_strength > 0 else 0
        
        candidate_mixes.append(mix_constrained[0])
        candidate_errors.append(error)
        candidate_percent_errors.append(percent_error)
    
    # 筛选误差最小的num_mixes个
    candidate_mixes = np.array(candidate_mixes)
    candidate_errors = np.array(candidate_errors)
    candidate_percent_errors = np.array(candidate_percent_errors)
    
    sorted_indices = np.argsort(candidate_errors)[:num_mixes]
    final_mixes = candidate_mixes[sorted_indices]
    final_errors = candidate_errors[sorted_indices]
    final_percent_errors = candidate_percent_errors[sorted_indices]
    
    # 转换为DataFrame
    mixes_df = pd.DataFrame(final_mixes, columns=FEATURE_COLUMNS_ORIGINAL)
    mixes_df['Predicted_CS_MPa'] = np.round(target_strength + (np.array(final_errors) * np.where(final_errors >= 0, 1, -1)), 2)
    mixes_df['Target_CS_MPa'] = target_strength
    mixes_df['Error_MPa'] = np.round(final_errors * np.where(final_errors >= 0, 1, -1), 2)
    mixes_df['Percentage_Error_%'] = np.round(final_percent_errors, 2)
    
    return mixes_df

# ====================== 低碳筛选函数 ======================
def filter_low_carbon_mixes(original_mixes_df):
    """从已生成的配合比中筛选低碳配合比"""
    # 计算每条配合比的碳排放
    original_mixes_df['碳排放(kg CO2 eq/m³)'] = original_mixes_df.apply(calculate_emission, axis=1)
    
    # 筛选碳排放≤阈值的配合比
    low_carbon_mixes = original_mixes_df[original_mixes_df['碳排放(kg CO2 eq/m³)'] <= CARBON_THRESHOLD].copy()
    
    if len(low_carbon_mixes) == 0:
        st.warning(f"没有找到碳排放≤{CARBON_THRESHOLD}的配合比，将返回碳排放最低的前50%")
        low_carbon_mixes = original_mixes_df.nsmallest(max(1, len(original_mixes_df)//2), '碳排放(kg CO2 eq/m³)')
    
    # 计算碳减排指标
    low_carbon_mixes['碳减排量(kg CO2 eq/m³)'] = np.round(REFERENCE_EMISSION - low_carbon_mixes['碳排放(kg CO2 eq/m³)'], 4)
    low_carbon_mixes['碳减排百分比(%)'] = np.round(
        (low_carbon_mixes['碳减排量(kg CO2 eq/m³)'] / REFERENCE_EMISSION) * 100, 2
    )
    
    return low_carbon_mixes.sort_values('综合评分', ascending=True)

# ====================== Streamlit网页主逻辑 ======================
def main():
    st.title("混凝土配合比逆向生成系统（含低碳筛选）")
    st.markdown("### 功能流程：1. 逆向生成配合比 → 2. 筛选低碳配合比")
    
    # 步骤1：加载数据和训练模型（首次运行时执行）
    with st.spinner("加载数据并训练模型..."):
        preprocessed_data = load_and_preprocess_data()
        models = train_models(preprocessed_data)
    
    # 步骤2：用户输入参数
    st.sidebar.header("参数设置")
    target_strength = st.sidebar.number_input(
        "目标强度 (MPa)",
        min_value=float(preprocessed_data['df'][preprocessed_data['target_column']].min() * 0.8),
        max_value=float(preprocessed_data['df'][preprocessed_data['target_column']].max() * 1.2),
        value=40.0,
        step=1.0
    )
    num_mixes = st.sidebar.slider(
        "生成配合比数量",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )
    
    # 步骤3：逆向生成配合比（无碳排放约束）
    if 'original_mixes' not in st.session_state:
        st.session_state['original_mixes'] = None
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 逆向生成配合比", type="primary"):
            with st.spinner(f"正在生成{num_mixes}个配合比..."):
                st.session_state['original_mixes'] = generate_mixes(
                    target_strength=target_strength,
                    num_mixes=num_mixes,
                    preprocessed_data=preprocessed_data,
                    models=models
                )
                st.success("配合比生成完成！")
    
    # 显示生成的配合比
    if st.session_state['original_mixes'] is not None:
        st.markdown("### 生成的配合比（无碳排放约束）")
        display_cols = ['OPC (kg/m3)', 'S (kg/m3)', 'FA (kg/m3)', 'W/B', 'Predicted_CS_MPa', 'Error_MPa', 'Percentage_Error_%']
        st.dataframe(st.session_state['original_mixes'][display_cols], use_container_width=True)
        
        # 步骤4：低碳筛选
        with col2:
            if st.button("🌱 筛选低碳配合比"):
                with st.spinner("正在筛选低碳配合比..."):
                    low_carbon_mixes = filter_low_carbon_mixes(st.session_state['original_mixes'])
                    st.session_state['low_carbon_mixes'] = low_carbon_mixes
                    st.success(f"筛选完成！共找到{len(low_carbon_mixes)}个低碳配合比")
        
        # 显示低碳配合比
        if 'low_carbon_mixes' in st.session_state:
            st.markdown("### 低碳配合比（碳排放≤600 kg CO2 eq/m³）")
            low_carbon_display_cols = display_cols + ['碳排放(kg CO2 eq/m³)', '碳减排量(kg CO2 eq/m³)', '碳减排百分比(%)']
            st.dataframe(st.session_state['low_carbon_mixes'][low_carbon_display_cols], use_container_width=True)
            
            # 下载功能
            csv = st.session_state['low_carbon_mixes'].to_csv(index=False)
            st.download_button(
                label="📥 下载低碳配合比",
                data=csv,
                file_name=f"低碳配合比_目标强度{target_strength}MPa.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
