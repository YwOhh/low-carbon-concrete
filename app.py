import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
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
    
    e_opc = row.get('OPC (kg/m3)', 0) * GWP['OPC (kg/m3)']
    e_s = row.get('S (kg/m3)', 0) * GWP['S (kg/m3)']
    e_fa = row.get('FA (kg/m3)', 0) * GWP['FA (kg/m3)']
    e_sf = row.get('SF (kg/m3)', 0) * GWP['SF (kg/m3)']
    e_gs = row.get('GS (kg/m3)', 0) * GWP['GS (kg/m3)']
    e_add = (row.get('SP (kg/m3)', 0) + row.get('HPMC (kg/m3)', 0)) * GWP['ADD']
    e_fiber = row.get('Fvol (%)f', 0) * GWP['Fvol (%)f']
    e_water = row.get('W (kg/m3)', 0) * GWP['W (kg/m3)']
    
    return e_opc + e_s + e_fa + e_sf + e_gs + e_add + e_fiber + e_water

# ====================== 数据加载与预处理 + 模型训练 ======================
@st.cache_resource
def load_and_preprocess_data():
    try:
        # ---------- 1. 读取 Excel ----------
        df = pd.read_excel('data.xlsx', sheet_name='3DP-FRC CS')
        st.success(f"✅ 数据加载成功！共 {df.shape[0]} 条记录，{df.shape[1]} 个字段")

        # ---------- 2. 列名规范化匹配 ----------
        def normalize(name):
            # 去除空格，将中文括号替换为英文括号，转为小写
            return name.replace(' ', '').replace('（', '(').replace('）', ')').lower()

        # 构建实际列名的规范化映射
        actual_columns = {normalize(col): col for col in df.columns}

        # 匹配特征列
        feature_columns = []
        for col in FEATURE_COLUMNS_ORIGINAL:
            norm_col = normalize(col)
            if norm_col in actual_columns:
                feature_columns.append(actual_columns[norm_col])
            else:
                st.warning(f"⚠ 缺失特征列：{col}，已跳过")

        # 匹配目标列
        target_column = None
        norm_target = normalize(TARGET_COLUMN_ORIGINAL)
        if norm_target in actual_columns:
            target_column = actual_columns[norm_target]
        else:
            st.error(f"❌ 缺失目标列：{TARGET_COLUMN_ORIGINAL}")
            st.stop()

        if not feature_columns:
            st.error("❌ 没有找到任何特征列，请检查列名配置")
            st.stop()

        # ---------- 3. 计算特征统计信息（用于约束） ----------
        feature_stats = {}
        for col in feature_columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                feature_stats[col] = {'min': 0, 'max': 100, 'non_zero_min': 0, 'mean': 0, 'std': 0}
                continue
            non_zero_data = col_data[col_data > 0]
            non_zero_min = non_zero_data.min() if len(non_zero_data) > 0 else 0

            # 使用分位数估计合理范围（避免异常值影响）
            q1 = col_data.quantile(0.05)
            q3 = col_data.quantile(0.95)
            iqr = q3 - q1
            min_val = max(0, q1 - 1.5 * iqr)
            max_val = q3 + 1.5 * iqr

            # 针对特定列调整范围
            if 'Fvol' in col.lower():
                min_val = max(min_val, non_zero_min * 0.5 if non_zero_min > 0 else 0.1)
                max_val = min(max_val, 10)
            elif 'FA' in col and 'kg/m3' in col:
                min_val = max(min_val, non_zero_min * 0.5 if non_zero_min > 0 else 20)
                max_val = min(max_val, 600)
            elif 'OPC' in col:
                min_val = max(min_val, non_zero_min * 0.5 if non_zero_min > 0 else 100)
                max_val = min(max_val, 800)
            elif 'W (kg/m3)' in col:
                min_val = max(min_val, non_zero_min * 0.5 if non_zero_min > 0 else 100)
            elif 'SP' in col or 'HPMC' in col:
                min_val = max(min_val, non_zero_min * 0.5 if non_zero_min > 0 else 0.1)

            feature_stats[col] = {
                'min': float(min_val),
                'max': float(max_val),
                'non_zero_min': float(non_zero_min),
                'mean': float(col_data.mean()),
                'std': float(col_data.std())
            }

        # ---------- 4. 数据划分：一半用于INN，一半用于ANN ----------
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_point = len(df_shuffled) // 2
        inn_data = df_shuffled.iloc[:split_point]   # 用于INN：强度 → 特征
        ann_data = df_shuffled.iloc[split_point:]   # 用于ANN：特征 → 强度

        # INN 数据：X = 强度, y = 特征
        inn_X = inn_data[[target_column]].values
        inn_y = inn_data[feature_columns].values
        inn_X_scaler = StandardScaler()
        inn_y_scaler = StandardScaler()
        inn_X_scaled = inn_X_scaler.fit_transform(inn_X)
        inn_y_scaled = inn_y_scaler.fit_transform(inn_y)

        # ANN 数据：X = 特征, y = 强度
        ann_X = ann_data[feature_columns].values
        ann_y = ann_data[[target_column]].values
        ann_X_scaler = StandardScaler()
        ann_y_scaler = StandardScaler()
        ann_X_scaled = ann_X_scaler.fit_transform(ann_X)
        ann_y_scaled = ann_y_scaler.fit_transform(ann_y)

        # ---------- 5. 训练模型 ----------
        # INN 模型：随机森林（输入强度，输出特征）
        inn_X_train, inn_X_test, inn_y_train, inn_y_test = train_test_split(
            inn_X_scaled, inn_y_scaled, test_size=0.2, random_state=42
        )
        inn_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        inn_model.fit(inn_X_train, inn_y_train)

        # ANN 模型：MLP（输入特征，输出强度）
        ann_X_train, ann_X_test, ann_y_train, ann_y_test = train_test_split(
            ann_X_scaled, ann_y_scaled, test_size=0.2, random_state=42
        )
        ann_model = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            max_iter=1000, random_state=42, learning_rate_init=0.001,
            alpha=0.0001, early_stopping=True, validation_fraction=0.1, verbose=False
        )
        ann_model.fit(ann_X_train, ann_y_train)

        # 评估
        inn_test_r2 = r2_score(inn_y_test, inn_model.predict(inn_X_test))
        ann_test_r2 = r2_score(ann_y_test, ann_model.predict(ann_X_test))
        st.success(f"✅ 模型训练完成！INN R²: {inn_test_r2:.4f}, ANN R²: {ann_test_r2:.4f}")

        # ---------- 6. 打包返回 ----------
        preprocessed_data = {
            'df': df,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'feature_stats': feature_stats,
            'inn_X_scaler': inn_X_scaler,
            'inn_y_scaler': inn_y_scaler,
            'ann_X_scaler': ann_X_scaler,
            'ann_y_scaler': ann_y_scaler,
            'inn_X_scaled': inn_X_scaled,
            'inn_y_scaled': inn_y_scaled,
            'ann_X_scaled': ann_X_scaled,
            'ann_y_scaled': ann_y_scaled,
            'models': {
                'inn_model': inn_model,
                'ann_model': ann_model,
                'inn_test_r2': inn_test_r2,
                'ann_test_r2': ann_test_r2
            }
        }
        return preprocessed_data

    except Exception as e:
        st.error(f"❌ 数据加载/预处理失败: {str(e)}")
        st.stop()

# ====================== 约束函数 ======================
def enforce_constraints(mixes_original, feature_columns, feature_stats, cd_value=28):
    mixes_original = np.maximum(mixes_original, 0)
    
    for i, col in enumerate(feature_columns):
        if col in feature_stats:
            mixes_original[:, i] = np.clip(mixes_original[:, i], feature_stats[col]['min'], feature_stats[col]['max'])
            
            if ('Fvol' in col.lower()) and feature_stats[col]['non_zero_min'] > 0:
                for j in range(len(mixes_original)):
                    if mixes_original[j, i] < feature_stats[col]['non_zero_min'] * 0.1:
                        mixes_original[j, i] = np.random.uniform(
                            feature_stats[col]['non_zero_min']*0.5, feature_stats[col]['non_zero_min']*2
                        )
            elif ('FA (kg/m3)' in col) and feature_stats[col]['non_zero_min'] > 0:
                for j in range(len(mixes_original)):
                    if mixes_original[j, i] < feature_stats[col]['non_zero_min'] * 0.1:
                        mixes_original[j, i] = np.random.uniform(
                            feature_stats[col]['non_zero_min']*0.5, feature_stats[col]['non_zero_min']*3
                        )
            elif ('OPC' in col) and feature_stats[col]['non_zero_min'] > 0:
                for j in range(len(mixes_original)):
                    if mixes_original[j, i] < feature_stats[col]['non_zero_min'] * 0.1:
                        mixes_original[j, i] = np.random.uniform(
                            feature_stats[col]['non_zero_min']*0.5, feature_stats[col]['non_zero_min']*2
                        )
    
    # 安全处理：只在列存在时赋值
    if 'CD（d)' in FEATURE_COLUMNS_ORIGINAL:
        cd_index = FEATURE_COLUMNS_ORIGINAL.index('CD（d)')
        if cd_index < mixes_original.shape[1]:
            mixes_original[:, cd_index] = cd_value
    
    if 'W/B' in FEATURE_COLUMNS_ORIGINAL:
        wb_index = FEATURE_COLUMNS_ORIGINAL.index('W/B')
        if wb_index < mixes_original.shape[1]:
            mixes_original[:, wb_index] = np.clip(mixes_original[:, wb_index], 0.2, 0.8)
    
    for i, col in enumerate(feature_columns):
        if 'Fvol' in col.lower():
            mixes_original[:, i] = np.round(mixes_original[:, i], 4)
        else:
            mixes_original[:, i] = np.round(mixes_original[:, i], 2)
    
    return np.maximum(mixes_original, 0)

# ====================== 配合比生成函数 ======================
def generate_mixes(target_strength, num_mixes, preprocessed_data, models):
    try:
        inn_model = models['inn_model']
        ann_model = models['ann_model']
        inn_X_scaler = preprocessed_data['inn_X_scaler']
        inn_y_scaler = preprocessed_data['inn_y_scaler']
        ann_X_scaler = preprocessed_data['ann_X_scaler']
        ann_y_scaler = preprocessed_data['ann_y_scaler']
        feature_columns = preprocessed_data['feature_columns']
        feature_stats = preprocessed_data['feature_stats']
        
        target_scaled = inn_X_scaler.transform([[target_strength]])
        candidate_mixes = []
        candidate_errors = []
        candidate_percent_errors = []
        
        with st.spinner("生成候选配合比..."):
            for i in range(num_mixes * 2):
                noise_level = 0.05 if i < num_mixes*2 else 0.1
                noise = np.random.normal(0, noise_level, size=target_scaled.shape)
                noisy_target = target_scaled + noise
                
                mix_scaled = inn_model.predict(noisy_target)
                mix_original = inn_y_scaler.inverse_transform(mix_scaled.reshape(1, -1))
                mix_constrained = enforce_constraints(mix_original, feature_columns, feature_stats)
                
                mix_scaled_constrained = ann_X_scaler.transform(mix_constrained)
                pred_scaled = ann_model.predict(mix_scaled_constrained)
                pred = ann_y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                
                error = abs(pred - target_strength)
                percent_error = (error / target_strength)*100 if target_strength>0 else 0
                
                candidate_mixes.append(mix_constrained[0])
                candidate_errors.append(error)
                candidate_percent_errors.append(percent_error)
        
        candidate_mixes = np.array(candidate_mixes)
        candidate_errors = np.array(candidate_errors)
        candidate_percent_errors = np.array(candidate_percent_errors)
        
        sorted_indices = np.argsort(candidate_errors)[:num_mixes]
        final_mixes = candidate_mixes[sorted_indices]
        final_errors = candidate_errors[sorted_indices]
        final_percent_errors = candidate_percent_errors[sorted_indices]
        
        mixes_df = pd.DataFrame(final_mixes, columns=FEATURE_COLUMNS_ORIGINAL)
        mixes_df['Predicted_CS_MPa'] = np.round(
            target_strength + (final_errors * np.where(final_errors >= 0, 1, -1)), 2
        )
        mixes_df['Target_CS_MPa'] = target_strength
        mixes_df['Error_MPa'] = np.round(final_errors * np.where(final_errors >= 0, 1, -1), 2)
        mixes_df['Percentage_Error_%'] = np.round(final_percent_errors, 2)
        
        return mixes_df
    except Exception as e:
        st.error(f"❌ 配合比生成失败：{str(e)}")
        return pd.DataFrame()

# ====================== 低碳筛选函数 ======================
def filter_low_carbon_mixes(original_mixes_df):
    try:
        if original_mixes_df.empty:
            st.warning("⚠ 无原始配合比可筛选")
            return pd.DataFrame()
        
        original_mixes_df['碳排放(kg CO2 eq/m³)'] = original_mixes_df.apply(calculate_emission, axis=1)
        low_carbon_mixes = original_mixes_df[original_mixes_df['碳排放(kg CO2 eq/m³)'] <= CARBON_THRESHOLD].copy()
        
        if len(low_carbon_mixes) == 0:
            st.warning(f"⚠ 无碳排放≤{CARBON_THRESHOLD}的配合比，返回碳排放最低的前50%")
            low_carbon_mixes = original_mixes_df.nsmallest(max(1, len(original_mixes_df)//2), '碳排放(kg CO2 eq/m³)')
        
        low_carbon_mixes['碳减排量(kg CO2 eq/m³)'] = np.round(REFERENCE_EMISSION - low_carbon_mixes['碳排放(kg CO2 eq/m³)'], 4)
        low_carbon_mixes['碳减排百分比(%)'] = np.round(
            (low_carbon_mixes['碳减排量(kg CO2 eq/m³)'] / REFERENCE_EMISSION)*100, 2
        )
        # 在这里创建综合评分
        low_carbon_mixes['综合评分'] = (
            low_carbon_mixes['Percentage_Error_%'] + 
            (low_carbon_mixes['碳排放(kg CO2 eq/m³)'] / CARBON_THRESHOLD) * 10
        )
        return low_carbon_mixes.sort_values('综合评分', ascending=True)
    except Exception as e:
        st.error(f"❌ 低碳筛选失败：{str(e)}")
        return pd.DataFrame()

# ====================== Streamlit网页主逻辑 ======================
def main():
    st.set_page_config(page_title="低碳混凝土配比", page_icon="🏗️")
    st.title("混凝土配合比逆向生成系统（含低碳筛选）")
    st.markdown("### 流程：逆向生成 → 低碳筛选")

    with st.spinner("加载数据 & 训练模型..."):
        preprocessed_data = load_and_preprocess_data()   # 现在包含了模型
        models = preprocessed_data['models']              # 提取模型

    st.sidebar.header("⚙️ 参数设置")
    min_strength = float(preprocessed_data['df'][preprocessed_data['target_column']].min() * 0.8)
    max_strength = float(preprocessed_data['df'][preprocessed_data['target_column']].max() * 1.2)
    target_strength = st.sidebar.number_input("目标强度 (MPa)", min_value=min_strength, max_value=max_strength, value=40.0, step=1.0)
    num_mixes = st.sidebar.slider("生成配合比数量", min_value=5, max_value=30, value=15, step=5)
    
    if 'original_mixes' not in st.session_state:
        st.session_state['original_mixes'] = pd.DataFrame()
    if 'low_carbon_mixes' not in st.session_state:
        st.session_state['low_carbon_mixes'] = pd.DataFrame()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 逆向生成配合比", type="primary"):
            st.session_state['original_mixes'] = generate_mixes(target_strength, num_mixes, preprocessed_data, models)
            st.session_state['low_carbon_mixes'] = pd.DataFrame()
    with col2:
        if st.button("🌱 筛选低碳配合比", disabled=st.session_state['original_mixes'].empty):
            st.session_state['low_carbon_mixes'] = filter_low_carbon_mixes(st.session_state['original_mixes'])
    
    if not st.session_state['original_mixes'].empty:
        st.markdown("---")
        st.markdown("### 📋 生成的配合比（无碳排放约束）")
        display_cols = ['OPC (kg/m3)', 'S (kg/m3)', 'FA (kg/m3)', 'W/B', 'Predicted_CS_MPa', 'Error_MPa', 'Percentage_Error_%']
        display_cols = [c for c in display_cols if c in st.session_state['original_mixes'].columns]
        st.dataframe(st.session_state['original_mixes'][display_cols], use_container_width=True)
    
    if not st.session_state['low_carbon_mixes'].empty:
        st.markdown("---")
        st.markdown("### 🌱 低碳配合比")
        low_carbon_display_cols = display_cols + ['碳排放(kg CO2 eq/m³)', '碳减排量(kg CO2 eq/m³)', '碳减排百分比(%)']
        low_carbon_display_cols = [c for c in low_carbon_display_cols if c in st.session_state['low_carbon_mixes'].columns]
        st.dataframe(st.session_state['low_carbon_mixes'][low_carbon_display_cols], use_container_width=True)
        
        csv = st.session_state['low_carbon_mixes'].to_csv(index=False, encoding='utf-8-sig')
        st.download_button("📥 下载CSV", csv, f"低碳配合比_{target_strength}MPa.csv", "text/csv")

if __name__ == "__main__":
    main()

