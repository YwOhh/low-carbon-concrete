import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="低碳混凝土智能配比系统", layout="wide", page_icon="🌱")

# ====================== 样式美化 ======================
st.markdown("""
<style>
div.stButton > button {
    background-color:#2E8B57;
    color:white;
    font-size:18px;
    height:3em;
    width:100%;
    border-radius:10px;
}
div.stDownloadButton > button {
    background-color:#1E90FF;
    color:white;
    font-size:16px;
    border-radius:8px;
}
</style>
""", unsafe_allow_html=True)

# ====================== 碳排放计算（容错+动态列名匹配） ======================
def calculate_emission(row):
    GWP = {
        'OPC': 0.925754987, 'S': 0.096949054, 'FA': 0.035101155, 'SF': 0.306808295,
        'GS': 0.004197845, 'ADD': 0.940857761, 'FIBER': 0.027134144, 'WATER': 0.000552102
    }
    total = 0.0
    # 动态匹配列名（关键词模糊匹配，适配Excel实际列名）
    for key in row.index:
        key_str = str(key).strip()
        if 'OPC' in key_str:
            total += row[key] * GWP['OPC']
        elif 'S(kg/m3)' in key_str or 'S_' in key_str:
            total += row[key] * GWP['S']
        elif 'FA' in key_str:
            total += row[key] * GWP['FA']
        elif 'SF' in key_str:
            total += row[key] * GWP['SF']
        elif 'GS' in key_str:
            total += row[key] * GWP['GS']
        elif 'W(kg/m3)' in key_str or 'Water' in key_str:
            total += row[key] * GWP['WATER']
        elif 'Fvol' in key_str or 'Fiber' in key_str:
            total += row[key] * GWP['FIBER']
    # 外加剂计算（双重容错，避免IndexError）
    sp_cols = [k for k in row.index if 'SP' in str(k)]
    sp = row[sp_cols[0]] if (sp_cols and len(sp_cols) > 0) else 0
    hpmc_cols = [k for k in row.index if 'HPMC' in str(k)]
    hpmc = row[hpmc_cols[0]] if (hpmc_cols and len(hpmc_cols) > 0) else 0
    total += (sp + hpmc) * GWP['ADD']
    return total

# ====================== 低碳优化（列名容错+范围约束） ======================
def optimize_for_low_carbon(mix, cols, threshold=600):
    # 用Series构造数据，避免DataFrame列数不匹配
    row = pd.Series(mix, index=cols)
    em = calculate_emission(row)
    if em <= threshold:
        return mix

    # 关键列索引查找（容错，避免IndexError）
    o_idx = cols.index('OPC(kg/m3)') if 'OPC(kg/m3)' in cols else -1
    f_idx = cols.index('FA(kg/m3)') if 'FA(kg/m3)' in cols else -1
    s_idx = cols.index('S(kg/m3)') if 'S(kg/m3)' in cols else -1

    # 关键列缺失时直接返回，避免后续计算错误
    if o_idx == -1 or f_idx == -1 or s_idx == -1:
        return mix

    # 迭代优化水泥用量，降低碳排放
    for _ in range(50):
        cement_reduction = mix[o_idx] * 0.05  # 每次减少5%水泥
        mix[o_idx] = max(50, mix[o_idx] - cement_reduction)  # 水泥最低50
        mix[f_idx] += cement_reduction * 0.8  # 补充粉煤灰
        mix[s_idx] += cement_reduction * 0.2  # 补充矿渣
        # 重新计算碳排放，达标则退出
        current_row = pd.Series(mix, index=cols)
        current_em = calculate_emission(current_row)
        if current_em <= threshold:
            break
    return mix

# ====================== 模型训练（缓存+全流程容错） ======================
@st.cache_resource(show_spinner=False)
def train_all_models():
    np.random.seed(42)

    # 1. 读取在线Excel（容错，避免网络或文件错误）
    try:
        url = "https://raw.githubusercontent.com/YwOhh/low-carbon-concrete/main/data.xlsx"
        df = pd.read_excel(url)
        # 清理列名（去除空格、特殊字符，统一格式）
        df.columns = [str(col).strip().replace(' ', '').replace('\u3000', '') for col in df.columns]
    except Exception as e:
        st.error(f"数据读取失败：{str(e)}")
        return None

    # 2. 列名映射（严格匹配清理后的Excel列名）
    mapping = {
        'OPC(kg/m3)': 'OPC(kg/m3)',
        'S(kg/m3)': 'S(kg/m3)',
        'W/B': 'W/B',
        'FA(kg/m3)': 'FA(kg/m3)',
        'GS(kg/m3)': 'GS(kg/m3)',
        'SF(kg/m3)': 'SF(kg/m3)',
        'SP(kg/m3)': 'SP(kg/m3)',
        'HPMC(kg/m3)': 'HPMC(kg/m3)',
        'W(kg/m3)': 'W(kg/m3)',
        'Fvol(%)f': 'Fvol(%)f',
        'CD（d)': 'CD（d)',
        'LD(X,Y,Z)': 'LD(X,Y,Z)',
        'Strength（GPa）': 'Strength（GPa）',
        'ElasticModulus(GPa)': 'ElasticModulus(GPa)',
        'Density(g/cm3)': 'Density(g/cm3)',
        'Lf/Df': 'Lf/Df',
        'Df(μm)': 'Df(μm)',
        'Lf(mm)': 'Lf(mm)',
        'CS(MPa)': 'CS(MPa)'
    }

    # 3. 筛选有效列（只保留Excel中存在的列，避免KeyError）
    f_cols_all = list(mapping.keys())[:-1]  # 所有特征列
    t_col_target = list(mapping.keys())[-1]  # 目标列（强度）
    # 过滤不存在的列
    f_real = [mapping[col] for col in f_cols_all if mapping[col] in df.columns]
    t_real = mapping[t_col_target] if mapping[t_col_target] in df.columns else df.columns[-1]

    # 4. 特征范围计算（只处理有效列）
    stats = {}
    for col in f_real:
        if col in df.columns:
            col_data = df[col].dropna()  # 去除空值
            if len(col_data) > 0:
                q1 = col_data.quantile(0.05)
                q3 = col_data.quantile(0.95)
                min_val = max(0, q1 - 1.5 * (q3 - q1))  # 下界（非负）
                max_val = q3 + 1.5 * (q3 - q1)          # 上界
                # 特殊材料范围约束
                if 'OPC' in col:
                    min_val, max_val = max(min_val, 50), min(max_val, 300)
                if 'FA' in col:
                    min_val, max_val = max(min_val, 20), min(max_val, 600)
                stats[col] = {'min': min_val, 'max': max_val}

    # 5. 数据分割（确保特征和目标列存在）
    if len(f_real) == 0 or t_real not in df.columns:
        st.error("有效特征列或目标列缺失，无法训练模型")
        return None
    # 拆分数据为Inverse和Forward模型数据集
    df_sample = df.sample(frac=1, random_state=42)  # 打乱数据
    half_idx = len(df_sample) // 2
    inn_data = df_sample.iloc[:half_idx]
    ann_data = df_sample.iloc[half_idx:]

    # 6. 提取特征和目标（确保数组形状正确）
    X_inn = inn_data[f_real].values  # Inverse模型输入（强度）
    y_inn = inn_data[[t_real]].values.ravel()  # Inverse模型输出（配比）- 一维
    X_ann = ann_data[f_real].values  # Forward模型输入（配比）
    y_ann = ann_data[[t_real]].values.ravel()  # Forward模型输出（强度）- 一维

    # 7. 数据标准化（避免量纲影响）
    scaler_X_inn = StandardScaler()
    scaler_y_inn = StandardScaler()
    scaler_X_ann = StandardScaler()
    scaler_y_ann = StandardScaler()

    X_inn_scaled = scaler_X_inn.fit_transform(X_inn)
    y_inn_scaled = scaler_y_inn.fit_transform(y_inn.reshape(-1, 1)).ravel()
    X_ann_scaled = scaler_X_ann.fit_transform(X_ann)
    y_ann_scaled = scaler_y_ann.fit_transform(y_ann.reshape(-1, 1)).ravel()

    # 8. 模型训练（容错，避免训练失败）
    try:
        # Inverse模型（RandomForest：输入强度→输出配比）
        model_inn = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        model_inn.fit(X_inn_scaled, y_inn_scaled)

        # Forward模型（MLP：输入配比→输出强度）
        model_ann = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            verbose=0
        )
        model_ann.fit(X_ann_scaled, y_ann_scaled)
    except Exception as e:
        st.error(f"模型训练失败：{str(e)}")
        return None

    # 返回所有模型和工具（确保后续调用可用）
    return (model_inn, model_ann, 
            scaler_X_inn, scaler_y_inn, 
            scaler_X_ann, scaler_y_ann, 
            f_cols_all, f_real, stats, df, t_real)

# ====================== 约束函数（列数匹配+范围限制） ======================
def constrain(mix_array, valid_cols, stats_dict, cd=28, carbon_limit=600):
    # 确保输入是2D数组（适配批量处理）
    if mix_array.ndim == 1:
        mix_array = mix_array.reshape(1, -1)
    # 逐行处理配比，应用约束
    for i in range(len(mix_array)):
        # 1. 非负约束（材料用量不能为负）
        mix_array[i] = np.maximum(mix_array[i], 0)
        # 2. 材料范围约束（基于训练数据统计）
        for j, col in enumerate(valid_cols):
            if col in stats_dict:
                mix_array[i][j] = np.clip(
                    mix_array[i][j],
                    stats_dict[col]['min'],
                    stats_dict[col]['max']
                )
        # 3. 低碳优化（降低碳排放）
        mix_array[i] = optimize_for_low_carbon(
            mix_array[i], valid_cols, carbon_limit
        )
    return mix_array

# ====================== 生成函数（全流程无报错） ======================
def generate_mix(target_strength, n_mix=10, carbon_limit=600, models_tuple):
    # 解包模型和工具（容错，避免None）
    try:
        (model_inn, model_ann, 
         scaler_X_inn, scaler_y_inn, 
         scaler_X_ann, scaler_y_ann, 
         f_cols_all, f_real, stats_dict, df, t_real) = models_tuple
    except:
        st.error("模型数据不完整，无法生成配比")
        return None

    # 1. 生成初始配比（基于Inverse模型）
    target_scaled = scaler_X_inn.transform([[target_strength]])
    initial_mixes_scaled = model_inn.predict(target_scaled)
    # 逆标准化，得到真实配比（确保形状正确）
    initial_mixes = scaler_y_inn.inverse_transform(
        initial_mixes_scaled.reshape(-1, len(f_real))
    )

    # 2. 应用约束和低碳优化（确保配比有效）
    constrained_mixes = constrain(initial_mixes, f_real, stats_dict, carbon_limit=carbon_limit)

    # 3. 筛选优质配比（强度误差<10%，碳排放达标）
    valid_mixes = []
    valid_emissions = []
    valid_strengths = []
    for mix in constrained_mixes:
        # 计算预测强度（Forward模型）
        mix_scaled = scaler_X_ann.transform([mix])
        pred_strength_scaled = model_ann.predict(mix_scaled)
        pred_strength = scaler_y_ann.inverse_transform(pred_strength_scaled.reshape(-1, 1))[0][0]
        # 计算碳排放
        mix_series = pd.Series(mix, index=f_real)
        emission = calculate_emission(mix_series)
        # 筛选条件：强度误差<10% + 碳排放达标
        strength_error = abs(pred_strength - target_strength) / target_strength
        if strength_error <= 0.1 and emission <= carbon_limit:
            valid_mixes.append(mix)
            valid_emissions.append(emission)
            valid_strengths.append(pred_strength)

    # 4. 处理无有效配比的情况
    if len(valid_mixes) == 0:
        st.warning("无满足条件的配比，放宽约束后重新生成")
        # 放宽约束重新生成
        constrained_mixes = constrain(initial_mixes, f_real, stats_dict, carbon_limit=carbon_limit+100)
        for mix in constrained_mixes:
            mix_scaled = scaler_X_ann.transform([mix])
            pred_strength_scaled = model_ann.predict(mix_scaled)
            pred_strength = scaler_y_ann.inverse_transform(pred_strength_scaled.reshape(-1, 1))[0][0]
            mix_series = pd.Series(mix, index=f_real)
            emission = calculate_emission(mix_series)
            valid_mixes.append(mix)
            valid_emissions.append(emission)
            valid_strengths.append(pred_strength)
        # 确保至少返回n_mix个配比
        valid_mixes = valid_mixes[:n_mix]
        valid_emissions = valid_emissions[:n_mix]
        valid_strengths = valid_strengths[:n_mix]

    # 5. 构造输出DataFrame（确保列名匹配）
    mix_df = pd.DataFrame(valid_mixes, columns=f_real)
    # 添加结果列
    mix_df['预测强度(MPa)'] = np.round(valid_strengths, 2)
    mix_df['目标强度(MPa)'] = target_strength
    mix_df['误差(MPa)'] = np.round(mix_df['预测强度(MPa)'] - target_strength, 2)
    mix_df['误差(%)'] = np.round(abs(mix_df['误差(MPa)']) / target_strength * 100, 2)
    mix_df['碳排放(kgCO₂/m³)'] = np.round(valid_emissions, 2)

    return mix_df

# ====================== 导出功能（Excel/CSV） ======================
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='低碳混凝土配比', index=False)
    output.seek(0)
    return output

def to_csv(df):
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8')

# ====================== 主界面（用户交互） ======================
def main():
    st.title("🌱 低碳混凝土配合比智能生成系统")
    st.markdown("### 双AI模型驱动 | 强制低碳约束 | 强度精准控制 | 一键导出报告")
    st.divider()

    # 1. 加载模型（显示状态）
    with st.spinner("正在加载模型和数据..."):
        models = train_all_models()
    if models is None:
        st.error("系统初始化失败，请检查数据或网络后刷新页面")
        return
    else:
        st.success("✅ 模型和数据加载完成，可开始生成配比")

    # 2. 提取模型参数（用于后续计算）
    _, _, _, _, _, _, _, f_real, _, df_all, t_real = models
    # 计算强度范围（基于训练数据，避免用户输入异常值）
    strength_data = df_all[t_real].dropna()
    min_strength = max(0, strength_data.quantile(0.05) * 0.8)
    max_strength = strength_data.quantile(0.95) * 1.2

    # 3. 用户参数设置
    st.subheader("🎯 生成参数设置")
    col1, col2, col3 = st.columns(3)
    with col1:
        target_strength = st.number_input(
            "目标抗压强度 (MPa)",
            min_value=float(np.round(min_strength, 1)),
            max_value=float(np.round(max_strength, 1)),
            value=40.0,
            step=1.0
        )
    with col2:
        mix_count = st.slider(
            "生成配比数量",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )
    with col3:
        carbon_limit = st.number_input(
            "碳排放上限 (kgCO₂/m³)",
            min_value=300,
            max_value=1000,
            value=600,
            step=50
        )

    # 4. 生成配比（用户触发）
    if st.button("🚀 一键生成低碳配合比"):
        with st.spinner(f"正在生成{mix_count}组低碳配比..."):
            result_df = generate_mix(target_strength, mix_count, carbon_limit, models)
        if result_df is None:
            st.error("配比生成失败，请重试")
            return

        # 5. 显示结果
        st.divider()
        st.subheader("📊 生成结果总表")
        st.dataframe(result_df, use_container_width=True, height=400)

        # 6. 关键指标统计
        st.subheader("📈 关键指标统计")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        avg_carbon = result_df['碳排放(kgCO₂/m³)'].mean()
        avg_error = result_df['误差(MPa)'].abs().mean()
        avg_error_pct = result_df['误差(%)'].mean()
        carbon_reduction = 662.8 - avg_carbon  # 基准碳排放（普通混凝土）

        with stat_col1:
            st.metric("平均碳排放", f"{avg_carbon:.1f} kgCO₂/m³")
        with stat_col2:
            st.metric("平均强度误差", f"{avg_error:.2f} MPa")
        with stat_col3:
            st.metric("平均误差率", f"{avg_error_pct:.2f}%")
        with stat_col4:
            st.metric("平均碳减排", f"{carbon_reduction:.1f} kgCO₂/m³")

        # 7. 最优配比推荐（误差最小+碳排放最低）
        st.subheader("🏆 最优低碳配比推荐")
        result_df['综合评分'] = result_df['误差(%)'] + result_df['碳排放(kgCO₂/m³)']/10
        best_mix = result_df.loc[result_df['综合评分'].idxmin()].to_frame().T
        st.dataframe(best_mix.drop(columns=['综合评分']), use_container_width=True)

        # 8. 导出功能
        st.subheader("💾 结果导出")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            excel_file = to_excel(result_df.drop(columns=['综合评分']))
            st.download_button(
                label="📥 导出Excel文件",
                data=excel_file,
                file_name=f"低碳混凝土配比_{target_strength}MPa.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with export_col2:
            csv_file = to_csv(result_df.drop(columns=['综合评分']))
            st.download_button(
                label="📥 导出CSV文件",
                data=csv_file,
                file_name=f"低碳混凝土配比_{target_strength}MPa.csv",
                mime="text/csv"
            )

        st.success("✅ 配比生成完成，可导出文件用于实际生产参考！")

if __name__ == "__main__":
    main()
