import pandas as pd
import numpy as np
import streamlit as st
import base64
from io import BytesIO, StringIO
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

# ====================== 碳排放计算 ======================
def calculate_emission(row):
    GWP = {
        'OPC': 0.925754987, 'S': 0.096949054, 'FA': 0.035101155, 'SF': 0.306808295,
        'GS': 0.004197845, 'ADD': 0.940857761, 'FIBER': 0.027134144, 'WATER': 0.000552102
    }
    e = lambda k, v: row[k] * GWP[v] if k in row else 0
    total = (
        e('OPC (kg/m3)', 'OPC') + e('S (kg/m3)', 'S') + e('FA (kg/m3)', 'FA') +
        e('SF (kg/m3)', 'SF') + e('GS (kg/m3)', 'GS') +
        (row['SP (kg/m3)'] + row['HPMC (kg/m3)']) * GWP['ADD'] +
        e('Fvol (%)f', 'FIBER') + e('W (kg/m3)', 'WATER')
    )
    return total

# ====================== 低碳优化 ======================
def optimize_for_low_carbon(mix, cols, threshold=600):
    df = pd.DataFrame([mix], columns=cols)
    em = calculate_emission(df.iloc[0])
    if em <= threshold:
        return mix

    o = cols.index('OPC (kg/m3)')
    f = cols.index('FA (kg/m3)')
    s = cols.index('S (kg/m3)')

    for _ in range(50):
        red = mix[o] * 0.05
        mix[o] = max(50, mix[o] - red)
        mix[f] += red * 0.8
        mix[s] += red * 0.2
        df = pd.DataFrame([mix], columns=cols)
        em = calculate_emission(df.iloc[0])
        if em <= threshold:
            break
    return mix

# ====================== 模型训练（缓存） ======================
@st.cache_resource(show_spinner=False)
def train_all_models():
    np.random.seed(42)
    url = "https://raw.githubusercontent.com/YwOhh/low-carbon-concrete/main/data.xlsx"
    df = pd.read_excel(url)

    mapping = {
        'OPC (kg/m3)': 'OPC (kg/m3)', 'S (kg/m3)': 'S (kg/m3)', 'W/B': 'W/B',
        'FA (kg/m3)': 'FA (kg/m3)', 'GS (kg/m3)': 'GS (kg/m3)', 'SF (kg/m3)': 'SF (kg/m3)',
        'SP (kg/m3)': 'SP (kg/m3)', 'HPMC (kg/m3)': 'HPMC (kg/m3)', 'W (kg/m3)': 'W (kg/m3)',
        'Fvol (%)f': 'Fvol (%)f', 'CD（d)': 'CD（d)', 'LD (X,Y,Z)': 'LD (X,Y,Z)',
        'Strength （GPa）': 'Strength （GPa）', 'Elastic Modulus (GPa)': 'Elastic Modulus (GPa)',
        'Density (g/cm3)': 'Density (g/cm3)', 'Lf/Df': 'Lf/Df', 'Df (μm)': 'Df (μm)',
        'Lf (mm)': 'Lf (mm)', 'CS (MPa)': 'CS (MPa)'
    }

    f_cols = list(mapping.keys())[:-1]
    t_col = list(mapping.keys())[-1]
    f_real = [mapping[c] for c in f_cols]
    t_real = mapping[t_col]

    stats = {}
    for c in f_real:
        d = df[c]
        q1, q3 = d.quantile(0.05), d.quantile(0.95)
        mn, mx = max(0, q1 - 1.5*(q3-q1)), q3 + 1.5*(q3-q1)
        if 'OPC' in c: mn, mx = max(mn,50), min(mx,300)
        if 'FA' in c: mn, mx = max(mn,20), min(mx,600)
        stats[c] = {'min': mn, 'max': mx}

    d2 = df.sample(frac=1, random_state=42)
    half = len(d2)//2
    inn_d, ann_d = d2.iloc[:half], d2.iloc[half:]

    ix, iy = inn_d[[t_real]].values, inn_d[f_real].values
    ax, ay = ann_d[f_real].values, ann_d[[t_real]].values

    ix_s, iy_s = StandardScaler(), StandardScaler()
    ax_s, ay_s = StandardScaler(), StandardScaler()

    ix_n = ix_s.fit_transform(ix)
    iy_n = iy_s.fit_transform(iy)
    ax_n = ax_s.fit_transform(ax)
    ay_n = ay_s.fit_transform(ay)

    ix_tr, ix_te, iy_tr, iy_te = train_test_split(ix_n, iy_n, test_size=0.2, random_state=42)
    ax_tr, ax_te, ay_tr, ay_te = train_test_split(ax_n, ay_n, test_size=0.2, random_state=42)

    inn = RandomForestRegressor(300, max_depth=15, n_jobs=-1, random_state=42)
    ann = MLPRegressor((128,64,32), max_iter=2000, random_state=42, early_stopping=True)
    inn.fit(ix_tr, iy_tr)
    ann.fit(ax_tr, ay_tr)

    return inn, ann, ix_s, iy_s, ax_s, ay_s, f_cols, f_real, stats, df, t_real

# ====================== 约束 ======================
def constrain(mix, f_cols, stats, cd=28, carbon=600):
    mix = np.maximum(mix, 0)
    for i,c in enumerate(f_cols):
        if c in stats:
            mix[:,i] = np.clip(mix[:,i], stats[c]['min'], stats[c]['max'])
    for i in range(len(mix)):
        mix[i] = optimize_for_low_carbon(mix[i], f_cols, carbon)
    return mix

# ====================== 生成 ======================
def generate(target, n_mix, carbon, models):
    inn, ann, ix_s, iy_s, ax_s, ay_s, f_cols, f_real, stats, df, t_col = models
    tar_n = ix_s.transform([[target]])
    candidates, ems, errs = [], [], []

    for _ in range(1200):
        noise = np.random.normal(0, 0.04, tar_n.shape)
        m = inn.predict(tar_n + noise)
        m = iy_s.inverse_transform(m.reshape(1,-1))
        m = constrain(m, f_cols, stats, 28, carbon)
        candidates.append(m[0])

        e = calculate_emission(pd.DataFrame([m[0]], columns=f_cols).iloc[0])
        p = ay_s.inverse_transform(ann.predict(ax_s.transform(m)).reshape(-1,1))[0,0]
        ems.append(e)
        errs.append(abs(p-target))

    candidates = np.array(candidates)
    ems, errs = np.array(ems), np.array(errs)
    ok = np.where((ems <= carbon) & (errs/target <= 0.1))[0]
    if len(ok) == 0:
        ok = np.argsort(ems)[:n_mix]
    best = ok[np.argsort(errs[ok])[:n_mix]]

    out = pd.DataFrame(candidates[best], columns=f_cols)
    out['预测强度(MPa)'] = [ay_s.inverse_transform(ann.predict(ax_s.transform([c])).reshape(-1,1))[0,0] for c in candidates[best]]
    out['目标强度(MPa)'] = target
    out['误差(MPa)'] = out['预测强度(MPa)'] - target
    out['误差(%)'] = (out['误差(MPa)'].abs() / target *100).round(2)
    out['碳排放(kgCO₂/m³)'] = np.array(ems)[best].round(2)
    return out

# ====================== 导出 ======================
def to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='配比结果')
    out.seek(0)
    return out

def to_csv(df):
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

# ====================== 主界面 ======================
def main():
    st.title(" 低碳混凝土配合比智能生成系统")
    st.markdown("### 双AI模型 | 强制低碳 | 强度精准 | 一键导出报告")

    with st.spinner("模型加载中..."):
        models = train_all_models()
    st.success("✅ 模型加载完成，可以开始生成！")

    df_all = models[9]
    t_col = models[10]
    s_min, s_max = df_all[t_col].min(), df_all[t_col].max()

    st.divider()
    st.subheader("🎯 参数设置")
    c1,c2,c3 = st.columns(3)
    with c1:
        target = st.number_input("目标抗压强度 (MPa)", min_value=float(s_min*0.8), max_value=float(s_max*1.2), value=40.0, step=1.0)
    with c2:
        n = st.slider("生成配比数量", 1, 50, 10)
    with c3:
        carbon = st.number_input("碳排放上限 (kgCO₂/m³)", value=600, step=10)

    if st.button("🚀 一键生成低碳配合比"):
        with st.spinner("正在生成..."):
            res = generate(target, n, carbon, models)

        st.divider()
        st.subheader("📊 生成结果总表")
        st.dataframe(res, use_container_width=True, height=400)

        st.divider()
        st.subheader("📈 关键指标统计")
        a1,a2,a3,a4 = st.columns(4)
        with a1:
            st.metric("平均碳排放", f"{res['碳排放(kgCO₂/m³)'].mean():.1f}")
        with a2:
            st.metric("平均误差(MPa)", f"{res['误差(MPa)'].abs().mean():.2f}")
        with a3:
            st.metric("平均误差(%)", f"{res['误差(%)'].mean():.2f}%")
        with a4:
            st.metric("平均碳减排", f"{662.8 - res['碳排放(kgCO₂/m³)'].mean():.1f} kg")

        st.divider()
        st.subheader("🏆 最优低碳配比（误差最小+碳排放最低）")
        res['score'] = res['误差(MPa)'].abs() + res['碳排放(kgCO₂/m³)']/1000
        best = res.loc[res['score'].idxmin()]
        st.dataframe(pd.DataFrame(best).T, use_container_width=True)

        st.divider()
        st.subheader("💾 导出报告")
        d1,d2 = st.columns(2)
        with d1:
            xl = to_excel(res)
            st.download_button("📥 导出 Excel", xl, f"低碳配比_{target}MPa.xlsx", "application/vnd.ms-excel")
        with d2:
            csv_data = to_csv(res)
            st.download_button("📥 导出 CSV", csv_data, f"低碳配比_{target}MPa.csv", "text/csv")

        st.success("✅ 生成完成！可直接导出使用")

if __name__ == "__main__":

    main()

