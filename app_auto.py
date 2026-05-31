# ── HR Analytics — Automated Pipeline App ──────────────────────────────────
# Student: Ahmed Selim | Epsilon AI | Data Science | Instructor: Mostafa Sabry
# Run: streamlit run app_auto.py
# ────────────────────────────────────────────────────────────────────────────
import warnings, json, time, os
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, cross_val_predict, GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    f1_score
)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    TARGET_COL    = "left",
    RANDOM_STATE  = 42,
    CV_FOLDS      = 5,
    CV_SCORING    = "f1",
    PIPELINE_PATH = "full_pipeline.pkl",
    META_PATH     = "pipeline_meta.json",
)
ORDINAL_FEATURES = ["salary"]
SALARY_ORDER     = [["low","medium","high"]]
ENG_FEATURES     = ["work_intensity","satisfaction_eval_gap","long_stay_no_promo"]
PARAM_GRIDS = {
    "Logistic Regression" : {"classifier__C":[0.1,1.0,10.0]},
    "Decision Tree"       : {"classifier__max_depth":[5,8,12,None]},
    "Random Forest"       : {"classifier__n_estimators":[100,200],
                              "classifier__max_depth":[None,15,20]},
    "Gradient Boosting"   : {"classifier__n_estimators":[100,200],
                              "classifier__learning_rate":[0.05,0.1]},
    "K-Nearest Neighbors" : {"classifier__n_neighbors":[3,5,7,11]},
}

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="HR Auto Pipeline", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main{background:#F8FAFC}
.hero{background:linear-gradient(135deg,#0D1B2A,#1565C0);border-radius:14px;
      padding:1.8rem 2rem;color:#fff;margin-bottom:1.2rem}
.hero h1{color:#fff;font-size:1.9rem;margin:0}
.hero p{color:#BFDBFE;margin:.25rem 0 0;font-size:.9rem}
.kpi{background:#fff;border-radius:10px;padding:1rem;text-align:center;
     box-shadow:0 2px 8px rgba(0,0,0,.07);border-left:4px solid #1565C0}
.kpi-v{font-size:1.8rem;font-weight:800;color:#0D1B2A}
.kpi-l{font-size:.78rem;color:#64748B;margin-top:.2rem}
.pstep{background:#fff;border-radius:9px;padding:.8rem 1rem;margin:.3rem 0;
       box-shadow:0 1px 4px rgba(0,0,0,.06);border-left:4px solid #00B4D8}
.pred-hi{background:#FEF2F2;border:2px solid #EF4444;border-radius:12px;
         padding:1.3rem;text-align:center}
.pred-lo{background:#F0FDF4;border:2px solid #22C55E;border-radius:12px;
         padding:1.3rem;text-align:center}
</style>""", unsafe_allow_html=True)

# ── FeatureEngineer (must be defined before joblib.load) ──────────────────────
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        h = ("average_monthly_hours" if "average_monthly_hours" in X.columns
             else "average_montly_hours")
        X["work_intensity"]        = X["number_project"] * X[h] / 100
        X["satisfaction_eval_gap"] = X["last_evaluation"] - X["satisfaction_level"]
        X["long_stay_no_promo"]    = (
            (X["time_spend_company"] >= 4) & (X["promotion_last_5years"] == 0)
        ).astype(int)
        return X

# ── Pipeline builders ─────────────────────────────────────────────────────────
def build_preprocessor(num_feats, ord_feats, nom_feats):
    num_p = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  StandardScaler())])
    ord_p = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("enc", OrdinalEncoder(categories=SALARY_ORDER,
                              handle_unknown="use_encoded_value",unknown_value=-1))])
    nom_p = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("enc", OneHotEncoder(drop="first",handle_unknown="ignore",
                              sparse_output=False))])
    return ColumnTransformer(
        [("num",num_p,num_feats),("ord",ord_p,ord_feats),("nom",nom_p,nom_feats)],
        remainder="drop", verbose_feature_names_out=False)

def build_all_pipelines(preprocessor):
    rs = CFG["RANDOM_STATE"]
    clfs = {
        "Logistic Regression" : LogisticRegression(max_iter=1000,random_state=rs),
        "Decision Tree"       : DecisionTreeClassifier(max_depth=8,random_state=rs),
        "Random Forest"       : RandomForestClassifier(n_estimators=100,
                                    random_state=rs,n_jobs=-1),
        "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100,
                                    random_state=rs),
        "K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=5,n_jobs=-1),
    }
    return {
        name: Pipeline([("engineer",FeatureEngineer()),
                         ("preprocessor",preprocessor),
                         ("classifier",clf)])
        for name,clf in clfs.items()
    }

# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file_obj):
    df = pd.read_csv(file_obj)
    df.rename(columns={"sales":"Department",
                        "average_montly_hours":"average_monthly_hours"},inplace=True)
    df.drop_duplicates(inplace=True); df.reset_index(drop=True,inplace=True)
    return df

def detect_features(df):
    tgt     = CFG["TARGET_COL"]
    exclude = [tgt]+ORDINAL_FEATURES+ENG_FEATURES
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    nominal = [c for c in df.select_dtypes(include=["object"]).columns  if c not in exclude]
    return numeric, nominal

# ── Auto pipeline runner ──────────────────────────────────────────────────────
def run_auto_pipeline(df, pb, st_txt, tune=False):
    X = df.drop(columns=[CFG["TARGET_COL"]])
    y = df[CFG["TARGET_COL"]]
    numeric_feats, nominal_feats = detect_features(df)
    all_numeric = numeric_feats + ENG_FEATURES

    st_txt.text("🔧 [1/5] Building pipelines...")
    pb.progress(10)
    pre   = build_preprocessor(all_numeric, ORDINAL_FEATURES, nominal_feats)
    pipes = build_all_pipelines(pre)

    cv = StratifiedKFold(CFG["CV_FOLDS"],shuffle=True,random_state=CFG["RANDOM_STATE"])
    scoring = dict(accuracy="accuracy",f1="f1",precision="precision",
                   recall="recall",roc_auc="roc_auc")

    cv_results = {}
    for i,(name,pipe) in enumerate(pipes.items()):
        st_txt.text(f"♻️  [2/5] Cross-validating: {name}...")
        res = cross_validate(pipe,X,y,cv=cv,scoring=scoring,n_jobs=-1)
        cv_results[name] = {
            "F1"       : res["test_f1"].mean(),
            "F1_std"   : res["test_f1"].std(),
            "Accuracy" : res["test_accuracy"].mean(),
            "Precision": res["test_precision"].mean(),
            "Recall"   : res["test_recall"].mean(),
            "ROC-AUC"  : res["test_roc_auc"].mean(),
        }
        pb.progress(20 + (i+1)*11)

    best_name = max(cv_results, key=lambda k: cv_results[k]["F1"])

    st_txt.text(f"🎯 [3/5] Tuning [{best_name}]...")
    pb.progress(78)
    if tune and best_name in PARAM_GRIDS:
        gs = GridSearchCV(pipes[best_name],PARAM_GRIDS[best_name],
                          cv=cv,scoring=CFG["CV_SCORING"],n_jobs=-1,refit=True)
        gs.fit(X,y); final_pipe = gs.best_estimator_
    else:
        final_pipe = pipes[best_name]; final_pipe.fit(X,y)

    st_txt.text("📊 [4/5] Final CV scores...")
    pb.progress(88)
    final_scores = cross_validate(final_pipe,X,y,cv=cv,scoring=scoring,
                                   return_train_score=True,n_jobs=-1)

    st_txt.text("💾 [5/5] Saving pipeline...")
    pb.progress(95)
    joblib.dump(final_pipe, CFG["PIPELINE_PATH"])
    meta = dict(best_model=best_name,
                numeric_features=all_numeric,
                ordinal_features=ORDINAL_FEATURES,
                nominal_features=nominal_feats,
                target_col=CFG["TARGET_COL"],
                cv_metrics={
                    "Accuracy" : round(final_scores["test_accuracy"].mean(),4),
                    "F1"       : round(final_scores["test_f1"].mean(),4),
                    "Precision": round(final_scores["test_precision"].mean(),4),
                    "Recall"   : round(final_scores["test_recall"].mean(),4),
                    "ROC-AUC"  : round(final_scores["test_roc_auc"].mean(),4),
                })
    with open(CFG["META_PATH"],"w") as f: json.dump(meta,f,indent=2)
    pb.progress(100); st_txt.text("✅ Done!")
    return final_pipe, best_name, cv_results, final_scores, meta, X, y

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 HR Auto Pipeline")
    st.markdown("---")
    page = st.radio("Nav", [
        "🏠  Home",
        "📊  Data Explorer",
        "⚙️  Run Pipeline",
        "📋  Results & Evaluation",
        "🔮  Predict Employee",
        "📈  Insights",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**📁 Upload Dataset**")
    uploaded = st.file_uploader("HR_comma_sep.csv", type=["csv"])
    meta_ok  = os.path.exists(CFG["META_PATH"])
    model_ok = os.path.exists(CFG["PIPELINE_PATH"])
    if meta_ok:
        with open(CFG["META_PATH"]) as f: saved_meta = json.load(f)
        st.markdown("---")
        st.success(f"Saved: **{saved_meta['best_model']}**")
        m = saved_meta["cv_metrics"]
        st.caption(f"F1={m['F1']} · AUC={m['ROC-AUC']}")
    st.markdown("---")
    st.markdown("**👤 Ahmed Selim**")
    st.markdown("🏫 Epsilon AI · Data Science")
    st.markdown("👨‍🏫 Mostafa Sabry")

# ── Load data ──────────────────────────────────────────────────────────────────
df = None
if uploaded:
    df = load_data(uploaded)
elif os.path.exists("HR_comma_sep.csv"):
    with open("HR_comma_sep.csv","rb") as f: df = load_data(f)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown("""
    <div class="hero">
      <h1>🤖 HR Analytics — Automated ML Pipeline</h1>
      <p>FeatureEngineer → ColumnTransformer → 5-Fold CV → GridSearchCV → Auto Save</p>
      <p><strong>Ahmed Selim</strong> | Epsilon AI | Data Science | Instructor: Mostafa Sabry</p>
    </div>""", unsafe_allow_html=True)

    if df is not None:
        total = len(df); lp = df["left"].mean()*100
        c1,c2,c3,c4 = st.columns(4)
        for col,v,l in [(c1,f"{total:,}","👥 Employees"),
                        (c2,f"{lp:.1f}%","🚪 Attrition"),
                        (c3,str(df.shape[1]),"📊 Features"),
                        (c4,"Auto ✅","🤖 Pipeline")]:
            col.markdown(f'<div class="kpi"><div class="kpi-v">{v}</div>'
                         f'<div class="kpi-l">{l}</div></div>',unsafe_allow_html=True)

    st.markdown("---"); st.markdown("### 🏗️ Pipeline Architecture")
    steps = [
        ("1","load_and_inspect()","Load CSV → fix names → drop duplicates → detect features","#1565C0"),
        ("2","FeatureEngineer","Custom transformer: 3 new features — plugs into Pipeline","#7C3AED"),
        ("3","ColumnTransformer","Numeric: Impute+Scale · Ordinal: OrdinalEnc · Nominal: OneHotEnc","#D97706"),
        ("4","sklearn Pipeline","Engineer → Preprocessor → Classifier chained in one object","#059669"),
        ("5","cross_validate()","5-Fold StratifiedKFold — no manual train/test split","#DC2626"),
        ("6","GridSearchCV","Tunes best model hyperparameters inside the pipeline","#0891B2"),
        ("7","fit(X_all)","Final pipeline fitted on ALL data — no data wasted","#16A34A"),
        ("8","joblib.dump()","One .pkl file = Engineer + Preprocessor + Model","#64748B"),
    ]
    for num,title,desc,color in steps:
        st.markdown(
            f'<div class="pstep" style="border-left-color:{color}">'
            f'<span style="background:{color};color:#fff;border-radius:50%;'
            f'width:22px;height:22px;display:inline-flex;align-items:center;'
            f'justify-content:center;font-weight:800;font-size:.8rem;'
            f'margin-right:.5rem">{num}</span>'
            f'<strong style="color:{color}">{title}</strong> — '
            f'<span style="color:#64748B;font-size:.88rem">{desc}</span></div>',
            unsafe_allow_html=True)
    if df is None:
        st.warning("⚠️ Upload **HR_comma_sep.csv** in the sidebar to begin.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif "Data Explorer" in page:
    st.markdown("## 📊 Data Explorer")
    if df is None: st.warning("Upload dataset first."); st.stop()

    t1,t2,t3 = st.tabs(["📋 Preview","📈 Distributions","🔥 Correlations"])

    with t1:
        st.markdown(f"**{df.shape[0]:,} rows × {df.shape[1]} cols**")
        st.dataframe(df.head(30), use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Types**"); st.dataframe(df.dtypes.rename("Type"),use_container_width=True)
        with c2:
            st.markdown("**Stats**"); st.dataframe(df.describe().round(3),use_container_width=True)

    with t2:
        df_p = df.copy()
        df_p["Attrition"] = df_p["left"].map({0:"Stayed",1:"Left"})
        counts = df["left"].value_counts().reset_index()
        counts.columns=["Status","Count"]
        counts["Status"] = counts["Status"].map({0:"Stayed",1:"Left"})
        fig = make_subplots(rows=1,cols=2,
            subplot_titles=["Count","(%)"],specs=[[{"type":"bar"},{"type":"pie"}]])
        fig.add_trace(go.Bar(x=counts["Status"],y=counts["Count"],
            marker_color=["#2ecc71","#e74c3c"],text=counts["Count"],
            textposition="outside"),row=1,col=1)
        fig.add_trace(go.Pie(labels=counts["Status"],values=counts["Count"],
            marker_colors=["#2ecc71","#e74c3c"],textinfo="percent+label",hole=0.3),
            row=1,col=2)
        fig.update_layout(height=360,showlegend=False,template="plotly_white",
                          title_text="🎯 Target: left")
        st.plotly_chart(fig, use_container_width=True)

        num_f,_ = detect_features(df)
        sel = st.selectbox("Feature", num_f)
        fig2 = px.histogram(df_p,x=sel,color="Attrition",barmode="overlay",
            opacity=0.65,nbins=30,
            color_discrete_map={"Stayed":"#2ecc71","Left":"#e74c3c"},
            title=f"{sel} by Attrition",template="plotly_white",height=340)
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        num_df = df.select_dtypes(include=[np.number])
        corr = num_df.corr().round(2)
        mask = np.triu(np.ones_like(corr,dtype=bool),k=1)
        fig3 = px.imshow(corr.where(~mask),text_auto=True,
            color_continuous_scale="RdYlGn",zmin=-1,zmax=1,
            title="🔥 Correlation Heatmap",template="plotly_white",height=500)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif "Run Pipeline" in page:
    st.markdown("## ⚙️ Run Automated Pipeline")
    if df is None: st.warning("Upload dataset first."); st.stop()

    st.markdown("### 🎛️ Settings")
    c1,c2,c3 = st.columns(3)
    with c1:
        cv_folds = st.select_slider("CV Folds",[3,5,7,10],value=5)
        CFG["CV_FOLDS"] = cv_folds
    with c2:
        cv_met = st.selectbox("Selection Metric",
            ["f1","accuracy","roc_auc","precision","recall"])
        CFG["CV_SCORING"] = cv_met
    with c3:
        do_tune = st.toggle("🎯 GridSearch Tuning",value=False)

    st.markdown("---")
    pc = st.columns(5)
    for col,(ico,t,s) in zip(pc,[("📥","Load","Auto clean"),
                                   ("🔧","Feat Eng","3 features"),
                                   ("🧱","Preprocess","Scale+Encode"),
                                   ("♻️","5-Fold CV","No split"),
                                   ("💾","Save","Full pipeline")]):
        col.markdown(f"**{ico} {t}**"); col.caption(s)
    st.markdown("---")

    if st.button("🚀 Run Full Auto Pipeline", use_container_width=True, type="primary"):
        pb = st.progress(0); st_txt = st.empty(); t0 = time.time()
        try:
            res = run_auto_pipeline(df, pb, st_txt, tune=do_tune)
            fp,bn,cvr,fs,meta,X,y = res
            elapsed = time.time()-t0
            st.session_state.update(dict(
                pipeline=fp, best_name=bn, cv_results=cvr,
                final_scores=fs, meta=meta, X=X, y=y, pipeline_ran=True))
            st.success(f"✅ Done in **{elapsed:.1f}s** | Best: **{bn}** | "
                       f"CV F1 = **{meta['cv_metrics']['F1']}**")
            m = meta["cv_metrics"]
            for col,(k,v) in zip(st.columns(5),m.items()): col.metric(k,f"{v:.4f}")
            st.info("👉 Go to **Results & Evaluation** for charts.")
        except Exception as e:
            st.error(f"❌ Error: {e}"); raise

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif "Results" in page:
    st.markdown("## 📋 Results & Evaluation")

    if "pipeline_ran" in st.session_state:
        fp=st.session_state["pipeline"]; bn=st.session_state["best_name"]
        cvr=st.session_state["cv_results"]; fs=st.session_state["final_scores"]
        meta=st.session_state["meta"]; X=st.session_state["X"]; y=st.session_state["y"]
        has_live = True
    elif model_ok and meta_ok and df is not None:
        with open(CFG["META_PATH"]) as f: meta=json.load(f)
        bn=meta["best_model"]; has_live=False
        st.info(f"Showing saved results — **{bn}**. Re-run for live charts.")
    else:
        st.warning("Run the pipeline first."); st.stop()

    if has_live:
        st.markdown("### 🏆 Model Comparison (CV)")
        cv_df = pd.DataFrame(cvr).T.round(4)
        cv_df.index.name = "Model"
        cols_s = ["F1","Accuracy","Precision","Recall","ROC-AUC"]
        def hl(col): top=col.max(); return ["background-color:#DCFCE7;font-weight:700" if v==top else "" for v in col]
        st.dataframe(cv_df[cols_s].style.apply(hl,axis=0).format("{:.4f}"),use_container_width=True)

        plot_df = cv_df[cols_s].reset_index().melt(id_vars="Model",var_name="Metric",value_name="Score")
        fig = px.bar(plot_df,x="Model",y="Score",color="Metric",barmode="group",
            text=plot_df["Score"].round(3),
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="📊 CV Model Comparison",range_y=[0.7,1.01],
            template="plotly_white",height=420)
        fig.update_traces(texttemplate="%{text}",textposition="outside",textfont_size=7)
        fig.update_layout(xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### 📊 Final Pipeline — **{bn}**")
    m = meta["cv_metrics"]
    for col,(k,v) in zip(st.columns(5),m.items()): col.metric(k,f"{v:.4f}")

    if has_live:
        cv_obj = StratifiedKFold(CFG["CV_FOLDS"],shuffle=True,random_state=CFG["RANDOM_STATE"])
        with st.spinner("Computing CV predictions..."):
            y_prob = cross_val_predict(fp,X,y,cv=cv_obj,method="predict_proba",n_jobs=-1)[:,1]
            y_pred = cross_val_predict(fp,X,y,cv=cv_obj,n_jobs=-1)

        col1,col2 = st.columns(2)
        with col1:
            fpr,tpr,_ = roc_curve(y,y_prob); auc_v = roc_auc_score(y,y_prob)
            fig2=go.Figure()
            fig2.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",
                name=f"{bn} (AUC={auc_v:.4f})",line=dict(color="#2563EB",width=2.5)))
            fig2.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
                line=dict(color="gray",dash="dash",width=1.5)))
            fig2.update_layout(title="📈 ROC Curve (CV)",xaxis_title="FPR",
                yaxis_title="TPR",template="plotly_white",height=390)
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            cm=confusion_matrix(y,y_pred)
            cm_pct=(cm.astype("float")/cm.sum(axis=1)[:,np.newaxis]*100).round(1)
            labels=["Stayed","Left"]
            fig3=make_subplots(rows=1,cols=2,subplot_titles=["Counts","%"])
            for ci,(mat,tmpl,cs) in enumerate(
                    [(cm,"%{text}","Blues"),(cm_pct,"%{text}%","Greens")],1):
                fig3.add_trace(go.Heatmap(z=mat[::-1],x=labels,y=labels[::-1],
                    text=mat[::-1],texttemplate=tmpl,colorscale=cs,showscale=False),
                    row=1,col=ci)
            fig3.update_layout(title="🔲 Confusion Matrix",height=390,template="plotly_white")
            fig3.update_xaxes(title_text="Predicted"); fig3.update_yaxes(title_text="Actual")
            st.plotly_chart(fig3, use_container_width=True)

        clf = fp.named_steps["classifier"]
        if hasattr(clf,"feature_importances_"):
            st.markdown("### 🔍 Feature Importance")
            pre = fp.named_steps["preprocessor"]
            num_f,nom_f = detect_features(df if df is not None else X)
            all_n = num_f+ENG_FEATURES
            nom_names = list(pre.named_transformers_["nom"].named_steps["enc"]
                              .get_feature_names_out(nom_f))
            fi = pd.DataFrame({"Feature":all_n+ORDINAL_FEATURES+nom_names,
                                "Importance":clf.feature_importances_}
                              ).sort_values("Importance",ascending=True)
            fi["Tier"]=fi["Importance"].apply(
                lambda v:"High (>10%)" if v>0.10 else "Medium (5-10%)" if v>0.05 else "Low (<5%)")
            fig4=px.bar(fi,x="Importance",y="Feature",color="Tier",
                color_discrete_map={"High (>10%)":"#e74c3c",
                                    "Medium (5-10%)":"#3498db","Low (<5%)":"#94a3b8"},
                orientation="h",text=fi["Importance"].round(3),
                title=f"Feature Importance — {bn}",template="plotly_white",height=540)
            fig4.update_traces(texttemplate="%{text}",textposition="outside")
            st.plotly_chart(fig4, use_container_width=True)

        with st.expander("📋 Classification Report"):
            st.code(classification_report(y,y_pred,target_names=["Stayed","Left"]))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown("## 🔮 Predict Employee Attrition Risk")
    if "pipeline_ran" in st.session_state:
        pipe = st.session_state["pipeline"]
    elif model_ok:
        pipe = joblib.load(CFG["PIPELINE_PATH"])
    else:
        st.warning("Run the pipeline first."); st.stop()

    dept_list = (sorted(df["Department"].unique().tolist())
                 if df is not None and "Department" in df.columns
                 else ["IT","RandD","accounting","hr","management",
                       "marketing","product_mng","sales","support","technical"])

    st.info("Pipeline handles all preprocessing automatically — just enter raw values.")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**📊 Performance**")
        sat  = st.slider("Satisfaction Level",0.0,1.0,0.60,0.01)
        lev  = st.slider("Last Evaluation",   0.0,1.0,0.70,0.01)
        nprj = st.slider("Number of Projects",2,7,4)
    with c2:
        st.markdown("**⏰ Hours & Time**")
        hrs  = st.slider("Avg Monthly Hours",80,320,200)
        yrs  = st.slider("Years at Company", 1, 10,  3)
        acc  = st.selectbox("Work Accident?",[0,1],format_func=lambda x:"Yes" if x else "No")
    with c3:
        st.markdown("**🏢 HR Info**")
        promo = st.selectbox("Promoted Last 5y?",[0,1],format_func=lambda x:"Yes" if x else "No")
        dept  = st.selectbox("Department",dept_list)
        sal   = st.selectbox("Salary Level",["low","medium","high"])

    if st.button("🔮 Predict Now", use_container_width=True, type="primary"):
        # Raw input — pipeline handles feature engineering + preprocessing
        raw = pd.DataFrame([{
            "satisfaction_level"   : sat,
            "last_evaluation"      : lev,
            "number_project"       : nprj,
            "average_monthly_hours": hrs,
            "time_spend_company"   : yrs,
            "Work_accident"        : acc,
            "promotion_last_5years": promo,
            "Department"           : dept,
            "salary"               : sal,
        }])

        pred = pipe.predict(raw)[0]
        prob = pipe.predict_proba(raw)[0][1]

        st.markdown("---")
        if pred == 1:
            st.markdown(f'<div class="pred-hi"><h2>⚠️ HIGH RISK</h2>'
                        f'<h3>Likely to <span style="color:#EF4444">LEAVE</span></h3>'
                        f'<p style="font-size:1.4rem">Risk: <strong>{prob*100:.1f}%</strong></p>'
                        f'</div>', unsafe_allow_html=True)
            st.error("**Recommended actions:**\n- 💬 Schedule 1-on-1\n"
                     "- 💰 Review compensation\n- 🚀 Career development talk\n"
                     "- 🏖️ Reduce workload")
        else:
            st.markdown(f'<div class="pred-lo"><h2>✅ LOW RISK</h2>'
                        f'<h3>Likely to <span style="color:#22C55E">STAY</span></h3>'
                        f'<p style="font-size:1.4rem">Risk: <strong>{prob*100:.1f}%</strong></p>'
                        f'</div>', unsafe_allow_html=True)
            st.success("Employee appears engaged. Keep supporting their growth!")

        st.progress(float(prob))
        risk = ("🔴 Very High" if prob>0.75 else "🟠 High" if prob>0.5
                else "🟡 Medium" if prob>0.25 else "🟢 Low")
        st.caption(f"Risk Level: **{risk}**  ({prob*100:.1f}%)")

        with st.expander("🔍 Auto-Engineered Features"):
            eng_out = FeatureEngineer().transform(raw.copy())
            st.dataframe(eng_out[["work_intensity",
                                   "satisfaction_eval_gap",
                                   "long_stay_no_promo"]].round(4),
                         use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Insights" in page:
    st.markdown("## 📈 Business Insights Dashboard")
    if df is None: st.warning("Upload dataset first."); st.stop()

    df_p = df.copy()
    df_p["Attrition"] = df_p["left"].map({0:"Stayed",1:"Left"})
    dept_col = "Department" if "Department" in df.columns else "sales"
    h_col    = "average_monthly_hours" if "average_monthly_hours" in df.columns else "average_montly_hours"

    c1,c2 = st.columns(2)
    with c1:
        r = (df.groupby(dept_col)["left"].mean().mul(100).round(1).reset_index())
        r.columns=[dept_col,"Attrition Rate (%)"]
        r=r.sort_values("Attrition Rate (%)",ascending=False)
        fig1=px.bar(r,x=dept_col,y="Attrition Rate (%)",color="Attrition Rate (%)",
            color_continuous_scale="RdYlGn_r",text="Attrition Rate (%)",
            title="🏢 By Department",template="plotly_white",height=360)
        fig1.update_traces(texttemplate="%{text}%",textposition="outside")
        fig1.update_coloraxes(showscale=False)
        st.plotly_chart(fig1,use_container_width=True)
    with c2:
        rs=(df.groupby("salary")["left"].mean().mul(100).round(1).reset_index())
        rs.columns=["salary","Attrition Rate (%)"]
        rs["salary"]=pd.Categorical(rs["salary"],
            categories=["low","medium","high"],ordered=True)
        rs=rs.sort_values("salary")
        fig2=px.bar(rs,x="salary",y="Attrition Rate (%)",color="Attrition Rate (%)",
            color_continuous_scale="RdYlGn_r",text="Attrition Rate (%)",
            title="💰 By Salary",template="plotly_white",height=360)
        fig2.update_traces(texttemplate="%{text}%",textposition="outside")
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2,use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        st_df=df[df["left"]==0]; lf_df=df[df["left"]==1]
        fig3=go.Figure()
        fig3.add_trace(go.Scatter(x=st_df["satisfaction_level"],y=st_df[h_col],
            mode="markers",opacity=0.25,marker=dict(color="#22C55E",size=4),name="Stayed"))
        fig3.add_trace(go.Scatter(x=lf_df["satisfaction_level"],y=lf_df[h_col],
            mode="markers",opacity=0.35,marker=dict(color="#EF4444",size=4),name="Left"))
        fig3.update_layout(title="💡 Satisfaction vs Hours",
            xaxis_title="Satisfaction",yaxis_title="Monthly Hours",
            template="plotly_white",height=360)
        st.plotly_chart(fig3,use_container_width=True)
    with c4:
        fig4=px.histogram(df_p,x="time_spend_company",color="Attrition",
            barmode="overlay",opacity=0.65,nbins=10,
            color_discrete_map={"Stayed":"#22C55E","Left":"#EF4444"},
            title="📅 Tenure Distribution",template="plotly_white",height=360)
        st.plotly_chart(fig4,use_container_width=True)

    st.markdown("---"); st.markdown("### 💡 Key Takeaways")
    ic1,ic2,ic3=st.columns(3)
    with ic1: st.error("**🔴 Top Risk Factor**\nLow satisfaction + high evaluation = highest flight risk.")
    with ic2: st.warning("**🟠 Stagnation Effect**\n4-6 yrs with no promotion spikes attrition.")
    with ic3: st.success("**🟢 Retention Lever**\nHigh salary employees leave 4× less.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align:center;color:#94a3b8;font-size:.8rem'>"
            "🤖 HR Analytics Auto Pipeline | Ahmed Selim | Epsilon AI | Data Science"
            "</div>", unsafe_allow_html=True)
