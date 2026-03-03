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
CARBON_THRESHOLD = 600
REFERENCE_EMISSION = 662.8490

# ====================== 碳排放计算函数 ======================
def calculate_emission(row):
    GWP = {
        'OPC (kg/m3)': 0.925754987,
        'S (kg/m3)': 0.096949054,
        'FA (kg/m3)': 0.035101155,
        'SF (kg/m3)': 0.306808295,
        'GS (kg/m3)': 0.004197845,
        'W (kg/m3)': 0.000552102,
        'Fvol (%)f': 0.027134144,
        'ADD': 0.940857761
    }
    def normalize(name):
        return name.replace(' ', '').replace('（', '(').replace('）', ')').lower()
    norm_row = {normalize(col): val for col, val in row.items()}
    e_opc = norm_row.get(normalize('OPC (kg/m3)'), 0) * GWP['OPC (kg/m3)']
    e_s = norm_row.get(normalize('S (kg/m3)'), 0) * GWP['S (kg/m3)']
    e_fa = norm_row.get(normalize('FA (kg/m3)'), 0) * GWP['FA (kg/m3)']
    e_sf = norm_row.get(normalize('SF (kg/m3)'), 0) * GWP['SF (kg/m3)']
    e_gs = norm_row.get(normalize('GS (kg/m3)'), 0) * GWP['GS (kg/m3)']
    e_add = (norm_row.get(normalize('SP (kg/m3)'), 0) + norm_row.get(normalize('HPMC (kg/m3)'), 0)) * GWP['ADD']
    e_fiber = norm_row.get(normalize('Fvol (%)f'), 0) * GWP['Fvol (%)f']
    e_water = norm_row.get(normalize('W (kg/m3)'), 0) * GWP['W (kg/m3)']
    return e_opc + e_s + e_fa + e_sf + e_gs + e_add + e_fiber + e_water

# ====================== 数据加载与预处理 + 模型训练 ======================
@st.cache_resource
def load_and_preprocess_data():
    try:
        df = pd.read_excel('data.xlsx', sheet_name='3DP-FRC CS')
        st.success(f"✅ 数据加载成功！共 {df.shape[0]} 条记录，{df.shape[1]} 个字段")

        def normalize(name):
            return name.replace(' ', '').replace('（', '(').replace('）', ')').lower()
        actual_columns = {normalize(col): col for col in df.columns}

        feature_columns = []
        for col in FEATURE_COLUMNS_ORIGINAL:
            norm_col = normalize(col)
            if norm_col in actual_columns:
                feature_columns.append(actual_columns[norm_col])
            else:
                st.warning(f"⚠ 缺失特征列：{col}，已跳过")

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

        # 剔除异常目标值（CS <= 5）
        original_count = len(df)
        df = df[df[target_column] > 5]
        if len(df) < original_count:
            st.info(f"已剔除 {original_count - len(df)} 条强度≤5MPa的异常记录，剩余 {len(df)} 条有效数据。")

        # 特征统计（略，与原代码相同）
        feature_stats = {}
        for col in feature_columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                feature_stats[col] = {'min': 0, 'max': 100, 'non_zero_min': 0, 'mean': 0, 'std': 0}
                continue
            non_zero_data = col_data[col_data > 0]
            non_zero_min = non_zero_data.min() if len(non_zero_data) > 0 else 0
            q1 = col_data.quantile(0.05)
            q3 = col_data.quantile(0.95)
            iqr = q3 - q1
            min_val = max(0, q1 - 1.5 * iqr)
            max_val = q3 + 1.5 * iqr
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
                'min': float(min_val), 'max': float(max_val),
                'non_zero_min': float(non_zero_min),
                'mean': float(col_data.mean()), 'std': float(col_data.std())
            }

        # 数据划分
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_point = len(df_shuffled) // 2
        inn_data = df_shuffled.iloc[:split_point]
        ann_data = df_shuffled.iloc[split_point:]

        inn_X = inn_data[[target_column]].values
        inn_y = inn_data[feature_columns].values
        inn_X_scaler = StandardScaler()
        inn_y_scaler = StandardScaler()
        inn_X_scaled = inn_X_scaler.fit_transform(inn_X)
        inn_y_scaled = inn_y_scaler.fit_transform(inn_y)

        ann_X = ann_data[feature_columns].values
        ann_y = ann_data[[target_column]].values
        ann_X_scaler = StandardScaler()
        ann_y_scaler = StandardScaler()
        ann_X_scaled = ann_X_scaler.fit_transform(ann_X)
        ann_y_scaled = ann_y_scaler.fit_transform(ann_y)

        # 训练模型（增强版）
        inn_X_train, inn_X_test, inn_y_train, inn_y_test = train_test_split(
            inn_X_scaled, inn_y_scaled, test_size=0.2, random_state=42
        )
        inn_model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=3,
            min_samples_leaf=1, random_state=42, n_jobs=-1
        )
        inn_model.fit(inn_X_train, inn_y_train)

        ann_X_train, ann_X_test, ann_y_train, ann_y_test = train_test_split(
            ann_X_scaled, ann_y_scaled, test_size=0.2, random_state=42
        )
        ann_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
            max_iter=2000, random_state=42, learning_rate_init=0.001,
            alpha=0.0001, early_stopping=True, validation_fraction=0.1, verbose=False
        )
        ann_model.fit(ann_X_train, ann_y_train)

        inn_test_r2 = r2_score(inn_y_test, inn_model.predict(inn_X_test))
        ann_test_r2 = r2_score(ann_y_test, ann_model.predict(ann_X_test))
        st.success(f"✅ 模型训练完成！INN R²: {inn_test_r2:.4f}, ANN R²: {ann_test_r2:.4f}")

        preprocessed_data = {
            'df': df, 'feature_columns': feature_columns, 'target_column': target_column,
            'feature_stats': feature_stats,
            'inn_X_scaler': inn_X_scaler, 'inn_y_scaler': inn_y_scaler,
            'ann_X_scaler': ann_X_scaler, 'ann_y_scaler': ann_y_scaler,
            'inn_X_scaled': inn_X_scaled, 'inn_y_scaled': inn_y_scaled,
            'ann_X_scaled': ann_X_scaled, 'ann_y_scaled': ann_y_scaled,
            'models': {
                'inn_model': inn_model, 'ann_model': ann_model,
                'inn_test_r2': inn_test_r2, 'ann_test_r2': ann_test_r2
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
                            feature_stats[col]['non_zero_min'] * 0.5,
                            feature_stats[col]['non_zero_min'] * 2
                        )
            elif ('FA' in col) and feature_stats[col]['non_zero_min'] > 0:
                for j in range(len(mixes_original)):
                    if mixes_original[j, i] < feature_stats[col]['non_zero_min'] * 0.1:
                        mixes_original[j, i] = np.random.uniform(
                            feature_stats[col]['non_zero_min'] * 0.5,
                            feature_stats[col]['non_zero_min'] * 3
                        )
            elif ('OPC' in col) and feature_stats[col]['non_zero_min'] > 0:
                for j in range(len(mixes_original)):
                    if mixes_original[j, i] < feature_stats[col]['non_zero_min'] * 0.1:
                        mixes_original[j, i] = np.random.uniform(
                            feature_stats[col]['non_zero_min'] * 0.5,
                            feature_stats[col]['non_zero_min'] * 2
                        )
    cd_actual_index = None
    wb_actual_index = None
    for i, col in enumerate(feature_columns):
        if 'CD' in col or '龄期' in col:
            cd_actual_index = i
        if 'W/B' in col:
            wb_actual_index = i
    if cd_actual_index is not None:
        mixes_original[:, cd_actual_index] = cd_value
    if wb_actual_index is not None:
        mixes_original[:, wb_actual_index] = np.clip(mixes_original[:, wb_actual_index], 0.2, 0.8)
    for i, col in enumerate(feature_columns):
        if 'Fvol' in col.lower():
            mixes_original[:, i] = np.round(mixes_original[:, i], 4)
        else:
            mixes_original[:, i] = np.round(mixes_original[:, i], 2)
    return np.maximum(mixes_original, 0)

# ====================== 配合比生成函数 ======================
def generate_mixes(target_strength, num_mixes, preprocessed_data, models, error_threshold):
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
            for i in range(num_mixes * 5):
                noise_level = 0.02 if i < num_mixes * 5 else 0.05
                noise = np.random.normal(0, noise_level, size=target_scaled.shape)
                noisy_target = target_scaled + noise
                mix_scaled = inn_model.predict(noisy_target)
                mix_original = inn_y_scaler.inverse_transform(mix_scaled.reshape(1, -1))
                mix_constrained = enforce_constraints(mix_original, feature_columns, feature_stats)
                mix_scaled_constrained = ann_X_scaler.transform(mix_constrained)
                pred_scaled = ann_model.predict(mix_scaled_constrained)
                pred = ann_y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                error = abs(pred - target_strength)
                percent_error = (error / target_strength) * 100 if target_strength > 0 else 0
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

        mixes_df = pd.DataFrame(final_mixes, columns=feature_columns)
        mixes_df['Predicted_CS_MPa'] = np.round(
            target_strength + (final_errors * np.where(final_errors >= 0, 1, -1)), 2
        )
        mixes_df['Target_CS_MPa'] = target_strength
        mixes_df['Error_MPa'] = np.round(final_errors * np.where(final_errors >= 0, 1, -1), 2)
        mixes_df['Percentage_Error_%'] = np.round(final_percent_errors, 2)

        # 确保列为数值类型
        mixes_df['Percentage_Error_%'] = pd.to_numeric(mixes_df['Percentage_Error_%'], errors='coerce')

        # 按用户设定的阈值过滤
        filtered_df = mixes_df[mixes_df['Percentage_Error_%'] < error_threshold].copy()
        if filtered_df.empty:
            st.warning(f"⚠ 生成的配合比中无误差小于 {error_threshold}% 的合格结果，请尝试调整参数或放宽阈值。")
        return filtered_df
    except Exception as e:
        st.error(f"❌ 配合比生成失败：{str(e)}")
        return pd.DataFrame()

# ====================== 低碳筛选函数 ======================
def filter_low_carbon_mixes(original_mixes_df):
    try:
        if original_mixes_df.empty:
            return pd.DataFrame()
        original_mixes_df['碳排放(kg CO2 eq/m³)'] = original_mixes_df.apply(calculate_emission, axis=1)
        low_carbon_mixes = original_mixes_df[original_mixes_df['碳排放(kg CO2 eq/m³)'] <= CARBON_THRESHOLD].copy()
        if len(low_carbon_mixes) == 0:
            st.warning(f"⚠ 无碳排放≤{CARBON_THRESHOLD}的配合比，返回碳排放最低的前50%")
            low_carbon_mixes = original_mixes_df.nsmallest(max(1, len(original_mixes_df)//2), '碳排放(kg CO2 eq/m³)')
        low_carbon_mixes['碳减排量(kg CO2 eq/m³)'] = np.round(REFERENCE_EMISSION - low_carbon_mixes['碳排放(kg CO2 eq/m³)'], 4)
        low_carbon_mixes['碳减排百分比(%)'] = np.round((low_carbon_mixes['碳减排量(kg CO2 eq/m³)'] / REFERENCE_EMISSION)*100, 2)
        low_carbon_mixes['综合评分'] = low_carbon_mixes['Percentage_Error_%'] + (low_carbon_mixes['碳排放(kg CO2 eq/m³)'] / CARBON_THRESHOLD) * 10
        return low_carbon_mixes.sort_values('综合评分', ascending=True)
    except Exception as e:
        st.error(f"❌ 低碳筛选失败：{str(e)}")
        return pd.DataFrame()

# ====================== 主逻辑 ======================
def main():
    st.set_page_config(page_title="低碳混凝土配比", page_icon="🏗️")
    st.title("智材通")
    st.markdown("### 流程：逆向生成 → 低碳筛选")

    with st.spinner("加载数据 & 训练模型..."):
        preprocessed_data = load_and_preprocess_data()
        models = preprocessed_data['models']

    st.sidebar.header("⚙️ 参数设置")
    min_strength = float(preprocessed_data['df'][preprocessed_data['target_column']].min() * 0.8)
    max_strength = float(preprocessed_data['df'][preprocessed_data['target_column']].max() * 1.2)
    target_strength = st.sidebar.number_input("目标强度 (MPa)", min_value=min_strength, max_value=max_strength, value=40.0, step=1.0)
    num_mixes = st.sidebar.slider("生成配合比数量", min_value=5, max_value=30, value=15, step=5)
    # 可调节误差阈值
    error_threshold = st.sidebar.slider("最大允许误差 (%)", min_value=1, max_value=20, value=5, step=1)

    if 'original_mixes' not in st.session_state:
        st.session_state['original_mixes'] = pd.DataFrame()
    if 'low_carbon_mixes' not in st.session_state:
        st.session_state['low_carbon_mixes'] = pd.DataFrame()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 逆向生成配合比", type="primary"):
            st.session_state['original_mixes'] = generate_mixes(target_strength, num_mixes, preprocessed_data, models, error_threshold)
            st.session_state['low_carbon_mixes'] = pd.DataFrame()
    with col2:
        if st.button("🌱 筛选低碳配合比", disabled=st.session_state['original_mixes'].empty):
            st.session_state['low_carbon_mixes'] = filter_low_carbon_mixes(st.session_state['original_mixes'])

    if not st.session_state['original_mixes'].empty:
        st.markdown("---")
        st.markdown(f"### 📋 生成的配合比（误差 < {error_threshold}%，无碳排放约束）")
        st.dataframe(st.session_state['original_mixes'], use_container_width=True)
        st.write("**误差统计**")
        st.write(st.session_state['original_mixes']['Percentage_Error_%'].describe())

    if not st.session_state['low_carbon_mixes'].empty:
        st.markdown("---")
        st.markdown(f"### 🌱 低碳配合比（误差 < {error_threshold}%）")
        st.dataframe(st.session_state['low_carbon_mixes'], use_container_width=True)
        csv = st.session_state['low_carbon_mixes'].to_csv(index=False, encoding='utf-8-sig')
        st.download_button("📥 下载CSV", csv, f"低碳配合比_{target_strength}MPa.csv", "text/csv")

if __name__ == "__main__":
    main()

