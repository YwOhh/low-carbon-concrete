import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ====================== 碳排放计算函数 ======================
def calculate_emission(row):
    """
    计算单条配合比的碳排放量（kg CO2 eq/m³）
    参数：
        row: pandas.DataFrame的行对象，包含配合比所有列
    返回：
        total_emission: 总碳排放量
    """
    # 各组分GWP系数（kg CO2 eq/kg）
    GWP_OPC = 0.925754987  # 水泥（高碳排放，重点控制）
    GWP_S = 0.096949054    # 矿渣
    GWP_FA = 0.035101155   # 粉煤灰（低碳替代料）
    GWP_SF = 0.306808295   # 硅灰
    GWP_GS = 0.004197845   # 粗骨料
    GWP_ADD = 0.940857761  # 外加剂（SP+HPMC合并）
    GWP_FIBER = 0.027134144 # 纤维（体积分数%）
    GWP_WATER = 0.000552102 # 水

    # 分项计算（兼容列名映射，确保能找到对应列）
    e_opc = row['OPC (kg/m3)'] * GWP_OPC if 'OPC (kg/m3)' in row else 0
    e_s = row['S (kg/m3)'] * GWP_S if 'S (kg/m3)' in row else 0
    e_fa = row['FA (kg/m3)'] * GWP_FA if 'FA (kg/m3)' in row else 0
    e_sf = row['SF (kg/m3)'] * GWP_SF if 'SF (kg/m3)' in row else 0
    e_gs = row['GS (kg/m3)'] * GWP_GS if 'GS (kg/m3)' in row else 0
    e_add = (row['SP (kg/m3)'] + row['HPMC (kg/m3)']) * GWP_ADD if 'SP (kg/m3)' in row and 'HPMC (kg/m3)' in row else 0
    e_fiber = row['Fvol (%)f'] * GWP_FIBER if 'Fvol (%)f' in row else 0
    e_water = row['W (kg/m3)'] * GWP_WATER if 'W (kg/m3)' in row else 0

    # 总碳排放
    total_emission = e_opc + e_s + e_fa + e_sf + e_gs + e_add + e_fiber + e_water
    return total_emission

# ====================== 约束函数（无碳排放优化） ======================
def enforce_constraints_basic(mixes_original, cd_value=28):
    """
    基础约束函数（仅保证参数合理性，不做碳排放优化）
    """
    # 确保所有参数为非负
    mixes_original = np.maximum(mixes_original, 0)

    # 对每个特征应用范围约束
    for i, col in enumerate(feature_columns):
        if col in feature_stats:
            min_val = feature_stats[col]['min']
            max_val = feature_stats[col]['max']

            # 应用范围约束
            mixes_original[:, i] = np.clip(mixes_original[:, i], min_val, max_val)

            # 对特定参数确保不为零
            if ('Fvol' in col or 'fvol' in col.lower()) and feature_stats[col]['non_zero_min'] > 0:
                # Fvol (%)f: 确保不为零，保留4位小数
                for j in range(len(mixes_original)):
                    if mixes_original[j, i] < feature_stats[col]['non_zero_min'] * 0.1:
                        mixes_original[j, i] = np.random.uniform(
                            feature_stats[col]['non_zero_min'] * 0.5,
                            feature_stats[col]['non_zero_min'] * 2
                        )

            elif ('FA' in col and 'FA (kg/m3)' in col) and feature_stats[col]['non_zero_min'] > 0:
                # FA: 确保不为零
                for j in range(len(mixes_original)):
                    if mixes_original[j, i] < feature_stats[col]['non_zero_min'] * 0.1:
                        mixes_original[j, i] = np.random.uniform(
                            feature_stats[col]['non_zero_min'] * 0.5,
                            feature_stats[col]['non_zero_min'] * 5
                        )

            elif ('OPC' in col) and feature_stats[col]['non_zero_min'] > 0:
                # OPC: 确保不为零
                for j in range(len(mixes_original)):
                    if mixes_original[j, i] < feature_stats[col]['non_zero_min'] * 0.1:
                        mixes_original[j, i] = np.random.uniform(
                            feature_stats[col]['non_zero_min'] * 0.5,
                            feature_stats[col]['non_zero_min'] * 1.5
                        )

            elif ('W' in col and 'W (kg/m3)' in col) and feature_stats[col]['non_zero_min'] > 0:
                # 水: 确保不为零
                for j in range(len(mixes_original)):
                    if mixes_original[j, i] < feature_stats[col]['non_zero_min'] * 0.1:
                        mixes_original[j, i] = np.random.uniform(
                            feature_stats[col]['non_zero_min'] * 0.5,
                            feature_stats[col]['non_zero_min'] * 2
                        )

    # 设置CD值为指定值
    cd_index = None
    for idx, col in enumerate(feature_columns_original):
        if 'CD' in col or 'cd' in col.lower():
            cd_index = idx
            break

    if cd_index is not None:
        mixes_original[:, cd_index] = cd_value

    # 确保W/B（水胶比）在合理范围内 (0.2-0.8)
    wb_index = None
    for idx, col in enumerate(feature_columns_original):
        if 'W/B' in col or 'w/b' in col.lower():
            wb_index = idx
            break

    if wb_index is not None:
        mixes_original[:, wb_index] = np.clip(mixes_original[:, wb_index], 0.2, 0.8)

    # 根据小数位数要求进行四舍五入
    for i, col in enumerate(feature_columns):
        if 'Fvol' in col or 'fvol' in col.lower():
            mixes_original[:, i] = np.round(mixes_original[:, i], 4)
        else:
            mixes_original[:, i] = np.round(mixes_original[:, i], 2)

    # 再次确保没有负值
    mixes_original = np.maximum(mixes_original, 0)

    return mixes_original

# ====================== 阶段1：生成无碳排放约束的大量配合比 ======================
def generate_mixes_without_carbon(target_strength, num_candidates=500, cd_value=28, error_threshold=5.0):
    """
    生成大量不考虑碳排放的配合比（仅考虑强度误差）
    """
    print(f"生成{num_candidates}个候选配合比（仅考虑强度误差≤±{error_threshold}%）...")

    # 将目标强度标准化
    target_scaled = inn_X_scaler.transform([[target_strength]])

    candidate_mixes = []
    candidate_errors = []
    candidate_percentage_errors = []

    # 生成大量候选配合比
    for i in range(num_candidates):
        # 不同的噪声水平，增加多样性
        if i < num_candidates * 0.5:  # 50%小噪声
            noise_level = 0.03 * (1 + np.random.rand())
        elif i < num_candidates * 0.85:  # 35%中等噪声
            noise_level = 0.1 * (1 + np.random.rand())
        else:  # 15%大噪声
            noise_level = 0.2 * (1 + np.random.rand())

        # 添加噪声
        noise = np.random.normal(0, noise_level, size=target_scaled.shape)
        noisy_target = target_scaled + noise

        # 生成配合比
        mix_scaled = inn_model.predict(noisy_target)

        # 反标准化
        mix_original = inn_y_scaler.inverse_transform(mix_scaled.reshape(1, -1))

        # 应用基础约束（无碳排放优化）
        mix_constrained = enforce_constraints_basic(mix_original, cd_value)

        # 预测强度
        mix_scaled_constrained = ann_X_scaler.transform(mix_constrained)
        pred_scaled = ann_model.predict(mix_scaled_constrained)
        pred = ann_y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))

        # 计算绝对误差和百分比误差
        error = abs(pred[0, 0] - target_strength)
        percentage_error = abs((pred[0, 0] - target_strength) / target_strength * 100) if target_strength > 0 else 0

        candidate_mixes.append(mix_constrained[0])
        candidate_errors.append(error)
        candidate_percentage_errors.append(percentage_error)

    # 转换为数组
    candidate_mixes = np.array(candidate_mixes)
    candidate_errors = np.array(candidate_errors)
    candidate_percentage_errors = np.array(candidate_percentage_errors)

    print(f"候选配合比误差范围: {candidate_errors.min():.4f} - {candidate_errors.max():.4f} MPa")
    print(f"候选配合比百分比误差范围: {candidate_percentage_errors.min():.2f}% - {candidate_percentage_errors.max():.2f}%")

    # 筛选误差在阈值内的配合比
    within_error_threshold = candidate_percentage_errors <= error_threshold
    valid_indices = np.where(within_error_threshold)[0]

    print(f"符合±{error_threshold}%误差的候选配合比数量: {len(valid_indices)}")

    if len(valid_indices) == 0:
        print(f"警告: 没有找到误差≤{error_threshold}%的配合比，放宽到±10%...")
        within_10_percent = candidate_percentage_errors <= 10.0
        valid_indices = np.where(within_10_percent)[0]

    # 选择误差最小的配合比
    if len(valid_indices) > 0:
        valid_errors = candidate_errors[valid_indices]
        # 按误差排序，保留所有有效配合比（保证数量充足）
        sorted_indices = valid_indices[np.argsort(valid_errors)]
        selected_mixes = candidate_mixes[sorted_indices]
        selected_errors = candidate_errors[sorted_indices]
        selected_percentage_errors = candidate_percentage_errors[sorted_indices]
    else:
        # 如果都不满足，选择误差最小的num_candidates个
        sorted_indices = np.argsort(candidate_errors)
        selected_mixes = candidate_mixes[sorted_indices[:num_candidates]]
        selected_errors = candidate_errors[sorted_indices[:num_candidates]]
        selected_percentage_errors = candidate_percentage_errors[sorted_indices[:num_candidates]]

    # 创建DataFrame
    mixes_df = pd.DataFrame(selected_mixes, columns=feature_columns_original)

    # 确保CD值为指定值
    cd_col_name = None
    for col in mixes_df.columns:
        if 'CD' in col or 'cd' in col.lower():
            cd_col_name = col
            break

    if cd_col_name:
        mixes_df[cd_col_name] = cd_value

    # 重新预测强度
    mixes_scaled = ann_X_scaler.transform(mixes_df.values)
    pred_scaled = ann_model.predict(mixes_scaled)
    pred = ann_y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))

    # 添加预测强度、误差列
    mixes_df['Predicted_CS_MPa'] = np.round(pred, 2)
    mixes_df['Target_CS_MPa'] = target_strength
    mixes_df['Error_MPa'] = np.round(mixes_df['Predicted_CS_MPa'] - target_strength, 2)
    mixes_df['Percentage_Error_%'] = np.round(
        abs(mixes_df['Predicted_CS_MPa'] - target_strength) / target_strength * 100, 2) if target_strength > 0 else 0

    return mixes_df

# ====================== 阶段2：从已有配合比中筛选低碳结果 ======================
def filter_low_carbon_mixes(original_mixes_df, carbon_threshold=600, num_selected=20):
    """
    从已生成的配合比中筛选低碳结果
    参数：
        original_mixes_df: 阶段1生成的无碳排放约束的配合比DataFrame
        carbon_threshold: 碳排放阈值
        num_selected: 最终选择的低碳配合比数量
    返回：
        low_carbon_mixes_df: 筛选后的低碳配合比
    """
    print(f"\n从{len(original_mixes_df)}个配合比中筛选碳排放≤{carbon_threshold}的低碳配合比...")

    # 计算每个配合比的碳排放
    original_mixes_df['碳排放(kg CO2 eq/m³)'] = original_mixes_df.apply(calculate_emission, axis=1)

    # 筛选碳排放≤阈值的配合比
    low_carbon_candidates = original_mixes_df[original_mixes_df['碳排放(kg CO2 eq/m³)'] <= carbon_threshold].copy()

    print(f"找到{len(low_carbon_candidates)}个碳排放≤{carbon_threshold}的配合比")

    if len(low_carbon_candidates) == 0:
        print(f"警告: 没有找到碳排放≤{carbon_threshold}的配合比，选择碳排放最低的{num_selected}个")
        # 按碳排放升序排序，选最低的
        low_carbon_candidates = original_mixes_df.sort_values('碳排放(kg CO2 eq/m³)').head(num_selected).copy()
    elif len(low_carbon_candidates) > num_selected:
        # 按误差+碳排放综合排序，选择最优的num_selected个
        low_carbon_candidates['综合评分'] = low_carbon_candidates['Error_MPa'].abs() + (low_carbon_candidates['碳排放(kg CO2 eq/m³)'] / 1000)
        low_carbon_candidates = low_carbon_candidates.sort_values('综合评分').head(num_selected).copy()

    # 重置索引
    low_carbon_candidates = low_carbon_candidates.reset_index(drop=True)

    print(f"最终筛选出{len(low_carbon_candidates)}个低碳配合比")
    print(f"低碳配合比碳排放范围: {low_carbon_candidates['碳排放(kg CO2 eq/m³)'].min():.2f} - {low_carbon_candidates['碳排放(kg CO2 eq/m³)'].max():.2f} kg CO2 eq/m³")

    return low_carbon_candidates

# ====================== 主程序 ======================
# 设置随机种子以确保可重复性
np.random.seed(42)

# 读取Excel文件
file_path = r"D:\啦啦啦啦啦\修改后完整数据集2.xlsx"

# 读取数据
df = pd.read_excel(file_path)
print("数据读取成功!")
print(f"数据形状: {df.shape}")

# 手动映射列名
column_mapping = {
    'OPC (kg/m3)': 'OPC (kg/m3)',
    'S (kg/m3)': 'S (kg/m3)',
    'W/B': 'W/B',
    'FA (kg/m3)': 'FA (kg/m3)',
    'GS (kg/m3)': 'GS (kg/m3)',
    'SF (kg/m3)': 'SF (kg/m3)',
    'SP (kg/m3)': 'SP (kg/m3)',
    'HPMC (kg/m3)': 'HPMC (kg/m3)',
    'W (kg/m3)': 'W (kg/m3)',
    'Fvol (%)f': 'Fvol (%)f',
    'CD（d)': 'CD（d)',
    'LD (X,Y,Z)': 'LD (X,Y,Z)',
    'Strength （GPa）': 'Strength （GPa）',
    'Elastic Modulus (GPa)': 'Elastic Modulus (GPa)',
    'Density (g/cm3)': 'Density (g/cm3)',
    'Lf/Df': 'Lf/Df',
    'Df (μm)': 'Df (μm)',
    'Lf (mm)': 'Lf (mm)',
    'CS (MPa)': 'CS (MPa)'
}

# 检查并更新列名映射
for expected_col, actual_col in column_mapping.items():
    if actual_col not in df.columns:
        for df_col in df.columns:
            if expected_col.lower().replace(' ', '').replace('（', '(').replace('）', ')') in df_col.lower().replace(' ', '').replace('（', '(').replace('）', ')'):
                column_mapping[expected_col] = df_col
                break

# 获取实际的特征列名和目标列名
feature_columns_original = list(column_mapping.keys())[:-1]  # 前18个是特征
target_column_original = list(column_mapping.keys())[-1]    # 最后一个是目标

feature_columns = [column_mapping[col] for col in feature_columns_original]
target_column = column_mapping[target_column_original]

print(f"\n使用的特征列 (18个):")
for i, col in enumerate(feature_columns):
    print(f"{i + 1:2d}. {col}")
print(f"\n目标列: {target_column}")

# 计算每个特征列的统计信息
feature_stats = {}
print("\n特征统计信息:")
for col in feature_columns:
    if col in df.columns:
        col_data = df[col]
        # 计算非零最小值（如果存在非零值）
        non_zero_data = col_data[col_data > 0]
        if len(non_zero_data) > 0:
            non_zero_min = non_zero_data.min()
        else:
            non_zero_min = 0

        # 计算合理的范围
        q1 = col_data.quantile(0.05)  # 5%分位数
        q3 = col_data.quantile(0.95)  # 95%分位数
        iqr = q3 - q1
        min_val = max(0, q1 - 1.5 * iqr)
        max_val = q3 + 1.5 * iqr

        # 特定参数的特殊处理
        if 'Fvol' in col or 'fvol' in col.lower():
            # 纤维体积率通常不为零，且需要较高精度
            if non_zero_min > 0:
                min_val = max(min_val, non_zero_min * 0.5)
            else:
                min_val = max(min_val, 0.1)  # 默认最小0.1%
            max_val = min(max_val, 10)  # 纤维体积率通常不超过10%

        if 'FA' in col and 'FA (kg/m3)' in col:
            # 粉煤灰
            if non_zero_min > 0:
                min_val = max(min_val, non_zero_min * 0.5)
            else:
                min_val = max(min_val, 20)  # 默认最小20 kg/m³
            max_val = min(max_val, 600)  # 提高粉煤灰上限

        if 'OPC' in col:
            # 水泥
            if non_zero_min > 0:
                min_val = max(min_val, non_zero_min * 0.5)
            else:
                min_val = max(min_val, 50)  # 最小50 kg/m³
            max_val = min(max_val, 300)  # 上限300 kg/m³

        if 'W' in col and 'W (kg/m3)' in col:
            # 水
            if non_zero_min > 0:
                min_val = max(min_val, non_zero_min * 0.5)
            else:
                min_val = max(min_val, 100)  # 默认最小100 kg/m³

        if 'SP' in col and 'SP (kg/m3)' in col:
            # 减水剂
            if non_zero_min > 0:
                min_val = max(min_val, non_zero_min * 0.5)
            else:
                min_val = max(min_val, 0.1)  # 默认最小0.1 kg/m³

        feature_stats[col] = {
            'min': float(min_val),
            'max': float(max_val),
            'non_zero_min': float(non_zero_min) if non_zero_min > 0 else 0,
            'mean': float(col_data.mean()),
            'std': float(col_data.std())
        }

# 分割数据
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_point = len(df_shuffled) // 2
inn_data = df_shuffled.iloc[:split_point].reset_index(drop=True)
ann_data = df_shuffled.iloc[split_point:].reset_index(drop=True)

print(f"\n数据分割:")
print(f"INN模型数据: {len(inn_data)} 行")
print(f"ANN模型数据: {len(ann_data)} 行")

# 准备数据
inn_X = inn_data[[target_column]].values
inn_y = inn_data[feature_columns].values
ann_X = ann_data[feature_columns].values
ann_y = ann_data[[target_column]].values

# 数据标准化
inn_X_scaler = StandardScaler()
inn_y_scaler = StandardScaler()
ann_X_scaler = StandardScaler()
ann_y_scaler = StandardScaler()

inn_X_scaled = inn_X_scaler.fit_transform(inn_X)
inn_y_scaled = inn_y_scaler.fit_transform(inn_y)
ann_X_scaled = ann_X_scaler.fit_transform(ann_X)
ann_y_scaled = ann_y_scaler.fit_transform(ann_y)

# 划分训练集和测试集
inn_X_train, inn_X_test, inn_y_train, inn_y_test = train_test_split(
    inn_X_scaled, inn_y_scaled, test_size=0.2, random_state=42
)

ann_X_train, ann_X_test, ann_y_train, ann_y_test = train_test_split(
    ann_X_scaled, ann_y_scaled, test_size=0.2, random_state=42
)

print(f"\nINN模型数据集:")
print(f"  训练集: {len(inn_X_train)} 个样本")
print(f"  测试集: {len(inn_X_test)} 个样本")

print(f"\nANN模型数据集:")
print(f"  训练集: {len(ann_X_train)} 个样本")
print(f"  测试集: {len(ann_X_test)} 个样本")

# 创建和训练INN模型
print("\n" + "=" * 60)
print("训练INN模型（逆模型：强度 -> 配合比）...")

inn_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

inn_model.fit(inn_X_train, inn_y_train)
print("INN模型训练完成!")

# 创建和训练ANN模型
print("\n" + "=" * 60)
print("训练ANN模型（前向模型：配合比 -> 强度）...")

ann_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42,
    learning_rate_init=0.001,
    alpha=0.0001,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)

ann_model.fit(ann_X_train, ann_y_train)
print("ANN模型训练完成!")

# ====================== 交互式生成和筛选 ======================
print("\n" + "=" * 60)
print("混凝土配合比生成系统（两阶段模式）")
print("=" * 60)
print("说明:")
print("1. 阶段1：生成大量不考虑碳排放的配合比（仅满足强度要求）")
print("2. 阶段2：点击低碳按钮，从已生成的配合比中筛选低碳结果")
print("=" * 60)

# 全局变量存储阶段1生成的配合比
global_mixes_df = None

while True:
    try:
        print(f"\n数据强度范围: {df[target_column].min():.2f} - {df[target_column].max():.2f} MPa")
        target_strength = float(input("请输入目标混凝土强度 (MPa) [输入0退出]: "))

        if target_strength == 0:
            print("感谢使用，再见！")
            break

        # 检查目标强度是否在合理范围内
        data_min = df[target_column].min()
        data_max = df[target_column].max()

        if target_strength < data_min * 0.8 or target_strength > data_max * 1.2:
            print(f"警告: 目标强度 {target_strength} MPa 超出数据范围的80-120% ({data_min:.2f} - {data_max:.2f} MPa)")
            proceed = input("是否继续? (y/n): ")
            if proceed.lower() != 'y':
                continue

        # 阶段1：生成大量无碳排放约束的配合比
        print(f"\n===== 阶段1：生成无碳排放约束的配合比 =====")
        num_candidates = int(input("请输入要生成的候选配合比数量 (默认500): ") or "500")
        error_threshold = float(input("请输入强度误差阈值(%) (默认5): ") or "5")
        
        print(f"\n正在生成{num_candidates}个配合比（仅考虑强度误差≤±{error_threshold}%）...")
        global_mixes_df = generate_mixes_without_carbon(
            target_strength, 
            num_candidates=num_candidates, 
            cd_value=28, 
            error_threshold=error_threshold
        )

        # 显示阶段1结果
        print("\n" + "=" * 120)
        print(f"阶段1完成：生成了{len(global_mixes_df)}个无碳排放约束的配合比")
        print("=" * 120)
        
        # 显示关键参数
        key_cols = ['OPC (kg/m3)', 'S (kg/m3)', 'FA (kg/m3)', 'W/B', 'W (kg/m3)', 
                   'Predicted_CS_MPa', 'Target_CS_MPa', 'Error_MPa', 'Percentage_Error_%']
        
        # 只显示前20个作为预览
        preview_df = global_mixes_df[key_cols].head(20).copy()
        pd.set_option('display.float_format', lambda x: f'{x:.4f}' if isinstance(x, (float, np.float64)) else str(x))
        print(f"\n配合比预览（前20个）:")
        print(preview_df.to_string(index=False))
        
        # 显示阶段1统计信息
        print(f"\n阶段1统计信息:")
        print(f"生成的配合比总数: {len(global_mixes_df)}")
        print(f"平均预测强度: {global_mixes_df['Predicted_CS_MPa'].mean():.4f} MPa")
        print(f"平均绝对误差: {global_mixes_df['Error_MPa'].abs().mean():.4f} MPa")
        print(f"平均百分比误差: {global_mixes_df['Percentage_Error_%'].mean():.4f}%")
        print(f"最大百分比误差: {global_mixes_df['Percentage_Error_%'].max():.4f}%")

        # 阶段2：低碳筛选
        low_carbon_choice = input("\n是否点击'低碳按钮'筛选低碳配合比? (y/n): ")
        if low_carbon_choice.lower() == 'y':
            print(f"\n===== 阶段2：筛选低碳配合比 =====")
            carbon_threshold = float(input("请输入碳排放阈值 (kg CO2 eq/m³) (默认600): ") or "600")
            num_selected = int(input("请输入最终选择的低碳配合比数量 (默认20): ") or "20")
            
            # 筛选低碳配合比
            low_carbon_df = filter_low_carbon_mixes(
                global_mixes_df, 
                carbon_threshold=carbon_threshold, 
                num_selected=num_selected
            )

            # 显示低碳筛选结果
            print("\n" + "=" * 120)
            print(f"低碳配合比筛选结果 (目标强度: {target_strength} MPa, 碳排放≤{carbon_threshold})")
            print("=" * 120)
            
            # 显示低碳配合比关键信息
            low_carbon_key_cols = key_cols + ['碳排放(kg CO2 eq/m³)']
            low_carbon_display_df = low_carbon_df[[col for col in low_carbon_key_cols if col in low_carbon_df.columns]].copy()
            print(f"\n低碳配合比列表:")
            print(low_carbon_display_df.to_string(index=False))

            # 低碳统计信息
            print(f"\n低碳配合比统计信息:")
            print(f"筛选出的低碳配合比数量: {len(low_carbon_df)}")
            print(f"平均碳排放量: {low_carbon_df['碳排放(kg CO2 eq/m³)'].mean():.4f} kg CO2 eq/m³")
            print(f"最低碳排放量: {low_carbon_df['碳排放(kg CO2 eq/m³)'].min():.4f} kg CO2 eq/m³")
            print(f"最高碳排放量: {low_carbon_df['碳排放(kg CO2 eq/m³)'].max():.4f} kg CO2 eq/m³")
            print(f"平均强度误差: {low_carbon_df['Percentage_Error_%'].mean():.4f}%")

            # 找到最优低碳配合比
            low_carbon_df['综合评分'] = low_carbon_df['Error_MPa'].abs() + (low_carbon_df['碳排放(kg CO2 eq/m³)'] / 1000)
            best_idx = low_carbon_df['综合评分'].argmin()
            best_mix = low_carbon_df.iloc[best_idx]

            print(f"\n最优低碳配合比 (序号 {best_idx + 1}):")
            print(f"  预测强度: {best_mix['Predicted_CS_MPa']:.4f} MPa")
            print(f"  强度误差: {best_mix['Percentage_Error_%']:.4f}%")
            print(f"  碳排放量: {best_mix['碳排放(kg CO2 eq/m³)']:.4f} kg CO2 eq/m³")

            # 保存低碳结果
            save_option = input("\n是否保存低碳配合比到Excel文件? (y/n): ")
            if save_option.lower() == 'y':
                save_path = f"D:\\啦啦啦啦啦\\低碳配合比_目标强度{target_strength}MPa_阈值{carbon_threshold}.xlsx"
                
                with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                    # 保存原始所有配合比
                    global_mixes_df.to_excel(writer, index=False, sheet_name='所有配合比')
                    # 保存低碳筛选结果
                    low_carbon_df.to_excel(writer, index=False, sheet_name='低碳配合比')
                    
                    # 设置列宽
                    for sheet_name in ['所有配合比', '低碳配合比']:
                        worksheet = writer.sheets[sheet_name]
                        df_sheet = global_mixes_df if sheet_name == '所有配合比' else low_carbon_df
                        for column in df_sheet:
                            column_width = max(df_sheet[column].astype(str).map(len).max(), len(column))
                            col_idx = df_sheet.columns.get_loc(column)
                            worksheet.column_dimensions[chr(65 + col_idx)].width = column_width + 2

                print(f"结果已保存到: {save_path}")

        # 保存阶段1结果
        save_original = input("\n是否保存所有生成的配合比（阶段1）到Excel文件? (y/n): ")
        if save_original.lower() == 'y':
            save_path = f"D:\\啦啦啦啦啦\\所有配合比_目标强度{target_strength}MPa.xlsx"
            
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                global_mixes_df.to_excel(writer, index=False, sheet_name='所有配合比')
                
                # 设置列宽
                worksheet = writer.sheets['所有配合比']
                for column in global_mixes_df:
                    column_width = max(global_mixes_df[column].astype(str).map(len).max(), len(column))
                    col_idx = global_mixes_df.columns.get_loc(column)
                    worksheet.column_dimensions[chr(65 + col_idx)].width = column_width + 2

            print(f"所有配合比已保存到: {save_path}")

    except ValueError as ve:
        print(f"输入错误: {ve}")
        print("请输入有效的数字！")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

# 模型评估
print("\n" + "=" * 60)
print("模型性能评估")
print("=" * 60)

# INN模型评估
inn_pred_train = inn_model.predict(inn_X_train)
inn_pred_test = inn_model.predict(inn_X_test)

inn_train_rmse = np.sqrt(mean_squared_error(inn_y_train, inn_pred_train))
inn_test_rmse = np.sqrt(mean_squared_error(inn_y_test, inn_pred_test))
inn_train_r2 = r2_score(inn_y_train, inn_pred_train)
inn_test_r2 = r2_score(inn_y_test, inn_pred_test)

print(f"INN模型性能 (强度 -> 配合比):")
print(f"  训练集RMSE: {inn_train_rmse:.6f} (标准化数据)")
print(f"  测试集RMSE: {inn_test_rmse:.6f} (标准化数据)")
print(f"  训练集R²: {inn_train_r2:.6f}")
print(f"  测试集R²: {inn_test_r2:.6f}")

# ANN模型评估
ann_pred_train = ann_model.predict(ann_X_train)
ann_pred_test = ann_model.predict(ann_X_test)

ann_train_rmse = np.sqrt(mean_squared_error(ann_y_train, ann_pred_train))
ann_test_rmse = np.sqrt(mean_squared_error(ann_y_test, ann_pred_test))
ann_train_r2 = r2_score(ann_y_train, ann_pred_train)
ann_test_r2 = r2_score(ann_y_test, ann_pred_test)

print(f"\nANN模型性能 (配合比 -> 强度):")
print(f"  训练集RMSE: {ann_train_rmse:.6f} (标准化数据)")
print(f"  测试集RMSE: {ann_test_rmse:.6f} (标准化数据)")
print(f"  训练集R²: {ann_train_r2:.6f}")
print(f"  测试集R²: {ann_test_r2:.6f}")

print("\n训练完成！")
print("程序结束。")
