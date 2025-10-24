
# Streamlit Nutri-Agente — MVP gratuito y escalable
# -------------------------------------------------
# Español: Código listo para correr con Streamlit.
# Objetivo: Agente conversacional simple para plan semanal, sugerencias
#           y lista de súper, alineado a reglas de porciones por grupo.
#
# Cómo usar:
# 1) Instala dependencias:  pip install streamlit pandas numpy
# 2) Ejecuta:               streamlit run app.py
# 3) Abre el navegador en la URL que imprime Streamlit.
#
# Notas de diseño (escalable):
# - Datos en memoria + st.session_state (rápido). Puedes migrar luego a SQLite/Postgres.
# - Estructuras alineadas al esquema propuesto (plans/rules/food_catalog/recipes/menus/menu_items).
# - "Chat" funciona sin LLM (reglas/heurísticas). Opcional: integrar Ollama local más adelante.

from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Nutri‑Agente", page_icon="🥗", layout="wide")

# =====================
# Utilidades y defaults
# =====================
MEAL_OPTS = ["desayuno", "colacion1", "comida", "colacion2", "cena"]
GROUPS = ["carb", "protein", "fat", "veg", "fruit", "dairy"]


def _default_rules() -> Dict:
    """Reglas base por comida: porciones por grupo y objetivos macro opcionales."""
    return {
        "plan_id": "PLAN_BASE",
        "name": "Plan Base",
        "portions_by_meal": {
            # Ajusta libremente. Tolerancia ±10% al validar.
            "desayuno": {"carb": 2, "protein": 1, "fat": 1, "veg": 0.5},
            "colacion1": {"carb": 1, "protein": 1, "fat": 0.5},
            "comida": {"carb": 2, "protein": 2, "fat": 1, "veg": 1},
            "colacion2": {"carb": 1, "protein": 1},
            "cena": {"carb": 1.5, "protein": 1.5, "fat": 1, "veg": 1},
        },
        # Objetivos macro por comida (opcional). Si no te sirven, déjalos vacíos.
        "macro_targets": {
            "desayuno": {"kcal": 350, "protein_g": 25},
            "comida": {"kcal": 600, "protein_g": 35},
            "cena": {"kcal": 450, "protein_g": 30},
        },
        "exclusions": [],  # Ej: ["lacteos", "cerdo"] — usamos tags en alimentos/recetas
    }


def _default_food_catalog() -> pd.DataFrame:
    data = [
        # id, name, group, portion_desc, portion_g, kcal, protein_g, carb_g, fat_g, tags
        ("F001", "Avena", "carb", "1/2 taza (40g)", 40, 150, 5, 27, 3, "integral"),
        ("F002", "Arroz integral cocido", "carb", "1 taza (160g)", 160, 216, 5, 45, 2, "integral"),
        ("F003", "Tortilla de maíz", "carb", "1 pieza (30g)", 30, 65, 2, 13, 1, "sin_gluten"),
        ("F010", "Pechuga de pollo", "protein", "100g", 100, 165, 31, 0, 3.6, "magro"),
        ("F011", "Claras de huevo", "protein", "4 claras (120g)", 120, 68, 15, 0.7, 0.2, "magro"),
        ("F012", "Atún en agua", "protein", "1 lata (120g)", 120, 132, 28, 0, 1, "económico"),
        ("F020", "Aguacate", "fat", "1/2 pieza (75g)", 75, 120, 1.5, 6, 10, "saludable"),
        ("F021", "Aceite de oliva", "fat", "1 cda (14g)", 14, 119, 0, 0, 14, ""),
        ("F022", "Nueces", "fat", "15g", 15, 100, 2.2, 2, 9.5, "snack"),
        ("F030", "Espinaca", "veg", "1 taza (30g)", 30, 7, 0.9, 1.1, 0.1, "fibra"),
        ("F031", "Brócoli", "veg", "1 taza (90g)", 90, 31, 2.5, 6, 0.3, "fibra"),
        ("F040", "Manzana", "fruit", "1 pza (150g)", 150, 78, 0.4, 21, 0.3, ""),
        ("F041", "Plátano", "fruit", "1 pza (120g)", 120, 105, 1.3, 27, 0.3, ""),
        ("F050", "Yoghurt griego natural", "dairy", "170g", 170, 100, 17, 6, 0, "alto_proteina"),
        ("F051", "Queso panela", "dairy", "40g", 40, 96, 7, 1, 6, ""),
    ]
    cols = [
        "food_id",
        "name",
        "group",
        "portion_desc",
        "portion_g",
        "kcal",
        "protein_g",
        "carb_g",
        "fat_g",
        "tags",
    ]
    return pd.DataFrame(data, columns=cols)


def _default_recipes() -> pd.DataFrame:
    data = [
        # id, name, servings, ingredients_json, instructions, tags
        (
            "R001",
            "Bowl de pollo con arroz y brócoli",
            2,
            json.dumps([
                {"food_id": "F010", "qty_portions": 2},  # 2x 100g pollo = 200g para 2 porciones
                {"food_id": "F002", "qty_portions": 2},  # 2 tazas arroz cocido total
                {"food_id": "F031", "qty_portions": 2},
                {"food_id": "F021", "qty_portions": 1},  # aceite al final
            ]),
            "Cocina pollo a la plancha, sirve con arroz y brócoli al vapor. Aliña con aceite.",
            "alto_proteina, preparado",
        ),
        (
            "R002",
            "Tostadas de atún con aguacate",
            1,
            json.dumps([
                {"food_id": "F012", "qty_portions": 1},
                {"food_id": "F003", "qty_portions": 2},
                {"food_id": "F020", "qty_portions": 0.5},
            ]),
            "Mezcla atún con aguacate y sirve sobre tortillas tostadas.",
            "rápido, sin_cocción",
        ),
        (
            "R003",
            "Avena con yoghurt, manzana y nueces",
            1,
            json.dumps([
                {"food_id": "F001", "qty_portions": 1},
                {"food_id": "F050", "qty_portions": 1},
                {"food_id": "F040", "qty_portions": 1},
                {"food_id": "F022", "qty_portions": 0.5},
            ]),
            "Hidrata la avena, mezcla con yoghurt y fruta, termina con nueces.",
            "desayuno",
        ),
    ]
    cols = ["recipe_id", "name", "servings", "ingredients_json", "instructions", "tags"]
    return pd.DataFrame(data, columns=cols)


# =====================
# Cálculo de macros
# =====================

def food_macros_per_portion(row: pd.Series) -> Dict[str, float]:
    return {
        "kcal": float(row["kcal"]),
        "protein_g": float(row["protein_g"]),
        "carb_g": float(row["carb_g"]),
        "fat_g": float(row["fat_g"]),
    }


def recipe_macros_per_serving(recipe_row: pd.Series, foods: pd.DataFrame) -> Dict[str, float]:
    try:
        ings = json.loads(recipe_row["ingredients_json"]) or []
    except Exception:
        ings = []
    totals = {"kcal": 0.0, "protein_g": 0.0, "carb_g": 0.0, "fat_g": 0.0}
    for ing in ings:
        f = foods.loc[foods["food_id"] == ing["food_id"]]
        if f.empty:
            continue
        macros = food_macros_per_portion(f.iloc[0])
        for k in totals:
            totals[k] += macros[k] * float(ing.get("qty_portions", 1))
    servings = max(1.0, float(recipe_row.get("servings", 1)))
    return {k: v / servings for k, v in totals.items()}


def sum_items_macros(items: List[Dict], foods: pd.DataFrame, recipes: pd.DataFrame) -> Dict[str, float]:
    totals = {"kcal": 0.0, "protein_g": 0.0, "carb_g": 0.0, "fat_g": 0.0}
    for it in items:
        if it["item_type"] == "food":
            f = foods.loc[foods["food_id"] == it["ref_id"]]
            if f.empty:
                continue
            macros = food_macros_per_portion(f.iloc[0])
            qty = float(it.get("servings_or_portions", 1))
            for k in totals:
                totals[k] += macros[k] * qty
        else:
            r = recipes.loc[recipes["recipe_id"] == it["ref_id"]]
            if r.empty:
                continue
            m = recipe_macros_per_serving(r.iloc[0], foods)
            qty = float(it.get("servings_or_portions", 1))
            for k in totals:
                totals[k] += m[k] * qty
    return totals


# =====================
# Sugerencias (heurísticas)
# =====================

def suggest_combo_for_meal(meal: str, rules: Dict, foods: pd.DataFrame, exclusions: List[str]) -> Dict:
    """Arma un combo simple (protein + carb + fat + veg/fruit) según porciones objetivo."""
    portions = rules.get("portions_by_meal", {}).get(meal, {})
    pick = {g: max(0.0, float(portions.get(g, 0))) for g in GROUPS}

    def pick_food(group: str) -> Optional[pd.Series]:
        df = foods[foods["group"] == group].copy()
        if exclusions:
            mask = ~df["tags"].fillna("").str.contains("|".join(exclusions), case=False)
            df = df[mask]
        if df.empty:
            return None
        # Heurística: prioriza proteína alta; carb integral; grasas saludables
        if group == "protein":
            df = df.sort_values("protein_g", ascending=False)
        elif group == "carb":
            df = df.sort_values("carb_g", ascending=False)
        elif group == "fat":
            df = df.sort_values("fat_g", ascending=False)
        else:
            df = df.sample(frac=1, random_state=42)
        return df.iloc[0]

    chosen: List[Dict] = []
    for g in GROUPS:
        qty = pick[g]
        if qty > 0:
            f = pick_food(g)
            if f is not None:
                chosen.append({
                    "item_type": "food",
                    "ref_id": f["food_id"],
                    "name": f["name"],
                    "meal": meal,
                    "servings_or_portions": qty,
                })
    return {"meal": meal, "items": chosen}


def suggest_recipes_for_meal(meal: str, recipes: pd.DataFrame, exclusions: List[str]) -> List[pd.Series]:
    df = recipes.copy()
    if exclusions:
        mask = ~df["tags"].fillna("").str.contains("|".join(exclusions), case=False)
        df = df[mask]
    # Heurística simple: por ahora, devolvemos hasta 3 recetas etiquetadas acorde
    hits = df[df["tags"].fillna("").str.contains(meal, case=False)]
    if hits.empty:
        hits = df
    return list(hits.head(3).itertuples(index=False))


# =====================
# Lista de súper
# =====================

def grocery_from_menu(menu_df: pd.DataFrame, foods: pd.DataFrame, recipes: pd.DataFrame) -> pd.DataFrame:
    """Agrega ingredientes de recetas y porciones de alimentos en el plan."""
    rows: List[Dict] = []
    for _, it in menu_df.iterrows():
        if it["item_type"] == "food":
            f = foods.loc[foods["food_id"] == it["ref_id"]]
            if f.empty:
                continue
            frow = f.iloc[0]
            rows.append({
                "food_id": frow["food_id"],
                "name": frow["name"],
                "unit": frow["portion_desc"],
                "qty": float(it["servings_or_portions"]),
                "group": frow["group"],
            })
        else:
            r = recipes.loc[recipes["recipe_id"] == it["ref_id"]]
            if r.empty():
                continue
            rrow = r.iloc[0]
            mult = float(it["servings_or_portions"])  # en porciones
            try:
                ings = json.loads(rrow["ingredients_json"]) or []
            except Exception:
                ings = []
            for ing in ings:
                f = foods.loc[foods["food_id"] == ing["food_id"]]
                if f.empty:
                    continue
                frow = f.iloc[0]
                rows.append({
                    "food_id": frow["food_id"],
                    "name": frow["name"],
                    "unit": frow["portion_desc"],
                    "qty": float(ing.get("qty_portions", 1)) * mult,
                    "group": frow["group"],
                })
    if not rows:
        return pd.DataFrame(columns=["food_id", "name", "unit", "qty", "group"])
    df = pd.DataFrame(rows)
    # Agregamos por food_id
    agg = df.groupby(["food_id", "name", "unit", "group"], as_index=False)["qty"].sum()
    # Redondeo amable para unidades de compra
    agg["qty_compra"] = agg["qty"].apply(lambda x: round(x + 1e-9, 2))
    return agg


# =====================
# Estado (Session)
# =====================
if "rules" not in st.session_state:
    st.session_state.rules = _default_rules()
if "foods" not in st.session_state:
    st.session_state.foods = _default_food_catalog()
if "recipes" not in st.session_state:
    st.session_state.recipes = _default_recipes()
if "menu" not in st.session_state:
    st.session_state.menu = pd.DataFrame(columns=[
        "date", "meal", "item_type", "ref_id", "name", "servings_or_portions",
    ])
if "chat" not in st.session_state:
    st.session_state.chat = []  # [{role: user/assistant, content: str}]


# =====================
# UI — Sidebar
# =====================
with st.sidebar:
    st.title("🥗 Nutri‑Agente")
    st.caption("MVP gratuito — reglas + heurísticas, sin LLM")
    st.markdown("**Plan activo:** ")
    st.text_input("Nombre del plan", key="plan_name", value=st.session_state.rules.get("name", "Plan Base"))
    st.text_input("Plan ID", key="plan_id", value=st.session_state.rules.get("plan_id", "PLAN_BASE"))

    st.markdown("---")
    st.markdown("**Persistencia**")
    if st.button("Descargar TODO (JSON)"):
        blob = {
            "rules": st.session_state.rules,
            "foods": st.session_state.foods.to_dict(orient="records"),
            "recipes": st.session_state.recipes.to_dict(orient="records"),
            "menu": st.session_state.menu.to_dict(orient="records"),
        }
        st.download_button(
            "Guardar backup.json",
            data=json.dumps(blob, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="nutri_agente_backup.json",
            mime="application/json",
        )

    uploaded = st.file_uploader("Cargar backup.json", type="json")
    if uploaded is not None:
        data = json.load(uploaded)
        st.session_state.rules = data.get("rules", st.session_state.rules)
        st.session_state.foods = pd.DataFrame(data.get("foods", [])) or st.session_state.foods
        st.session_state.recipes = pd.DataFrame(data.get("recipes", [])) or st.session_state.recipes
        st.session_state.menu = pd.DataFrame(data.get("menu", [])) or st.session_state.menu
        st.success("Backup cargado.")


# =====================
# UI — Tabs principales
# =====================
TAB_AYUDA, TAB_REGLAS, TAB_CATALOGO, TAB_RECETAS, TAB_MENU, TAB_SUPER, TAB_CHAT = st.tabs([
    "ℹ️ Ayuda", "⚙️ Reglas", "📚 Catálogo", "🍽️ Recetas", "🗓️ Plan semanal", "🛒 Lista de súper", "💬 Chat",
])


with TAB_AYUDA:
    st.header("Cómo funciona este MVP")
    st.markdown(
        """
        **Núcleo**: Defines reglas de porciones por comida (ej. desayuno = 2 carb, 1 proteína...).
        Cargas/edita tu catálogo de alimentos y recetas. En **Plan semanal** vas agregando lo que vas a comer
        y puedes generar la **Lista de súper**. En **Chat**, pídele sugerencias por comida.

        **Datos y escalabilidad**:
        - Este MVP guarda datos en memoria; exporta/importa "backup.json" para persistir.
        - Más adelante puedes migrar el esquema a SQLite/Postgres y agregar un LLM local (Ollama) o en la nube.
        - Estructuras compatibles con: `plans`, `rules`, `food_catalog`, `recipes`, `menus`, `menu_items`.
        """
    )

with TAB_REGLAS:
    st.subheader("Porciones por comida (por grupo)")
    # Convertimos reglas->DataFrame para edición sencilla
    pbm = st.session_state.rules.get("portions_by_meal", {})
    # Asegurar todas las comidas existan
    for m in MEAL_OPTS:
        pbm.setdefault(m, {})
    rules_df = pd.DataFrame.from_dict(pbm, orient="index").reindex(MEAL_OPTS)
    rules_df = rules_df.reindex(columns=GROUPS)
    edited = st.data_editor(rules_df.fillna(0.0), num_rows="dynamic")

    st.subheader("Objetivos macro por comida (opcional)")
    mt = st.session_state.rules.get("macro_targets", {})
    mt_df = pd.DataFrame.from_dict(mt, orient="index").reindex(MEAL_OPTS)
    edited_mt = st.data_editor(mt_df)

    st.subheader("Exclusiones (tags)")
    excl = st.text_input("Separadas por coma (ej. lacteos, cerdo)", value=", ".join(st.session_state.rules.get("exclusions", [])))

    if st.button("Guardar reglas"):
        # Actualizamos session_state.rules
        pbm_new = {m: {g: float(edited.loc[m, g]) for g in GROUPS if not pd.isna(edited.loc[m, g]) and float(edited.loc[m, g]) > 0}
                   for m in MEAL_OPTS}
        mt_new = {}
        for m in MEAL_OPTS:
            row = edited_mt.loc[m] if m in edited_mt.index else {}
            if isinstance(row, pd.Series):
                mt_new[m] = {k: float(v) for k, v in row.to_dict().items() if not pd.isna(v)}
        st.session_state.rules["portions_by_meal"] = pbm_new
        st.session_state.rules["macro_targets"] = mt_new
        st.session_state.rules["exclusions"] = [s.strip() for s in excl.split(",") if s.strip()]
        st.session_state.rules["name"] = st.session_state.plan_name
        st.session_state.rules["plan_id"] = st.session_state.plan_id
        st.success("Reglas actualizadas.")

with TAB_CATALOGO:
    st.subheader("Catálogo de alimentos (por 1 porción estándar)")
    st.caption("Edita libremente. Columnas requeridas: food_id, name, group, portion_desc, kcal, protein_g, carb_g, fat_g, tags")

    foods = st.session_state.foods.copy()
    foods = foods.reindex(columns=["food_id", "name", "group", "portion_desc", "portion_g", "kcal", "protein_g", "carb_g", "fat_g", "tags"])
    st.session_state.foods = st.data_editor(foods, num_rows="dynamic")

    st.download_button(
        "Descargar catalogo.csv",
        data=st.session_state.foods.to_csv(index=False).encode("utf-8"),
        file_name="catalogo_alimentos.csv",
        mime="text/csv",
    )

    up = st.file_uploader("Subir catalogo.csv", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        st.session_state.foods = df
        st.success("Catálogo cargado.")

with TAB_RECETAS:
    st.subheader("Recetas / Platillos")
    st.caption("ingredients_json = lista de objetos {food_id, qty_portions}")
    rec = st.session_state.recipes.copy()
    rec = rec.reindex(columns=["recipe_id", "name", "servings", "ingredients_json", "instructions", "tags"])
    st.session_state.recipes = st.data_editor(rec, num_rows="dynamic", height=400)

    # Vista rápida de macros por porción de la receta seleccionada
    st.markdown("**Vista rápida de macros**")
    idx = st.selectbox("Elige receta", options=["-"] + list(st.session_state.recipes["recipe_id"].astype(str)), index=0)
    if idx != "-":
        rrow = st.session_state.recipes.loc[st.session_state.recipes["recipe_id"] == idx]
        if not rrow.empty:
            m = recipe_macros_per_serving(rrow.iloc[0], st.session_state.foods)
            st.write(pd.DataFrame([m]))

    st.download_button(
        "Descargar recetas.csv",
        data=st.session_state.recipes.to_csv(index=False).encode("utf-8"),
        file_name="recetas.csv",
        mime="text/csv",
    )

    up2 = st.file_uploader("Subir recetas.csv", type=["csv"])
    if up2 is not None:
        df = pd.read_csv(up2)
        st.session_state.recipes = df
        st.success("Recetas cargadas.")

with TAB_MENU:
    st.subheader("Plan semanal (menu_items)")
    st.caption("Agrega entradas: date, meal, item_type (food/recipe), ref_id, name, servings_or_portions")

    # Atajo para crear semana base
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        start = st.date_input("Semana inicia", value=date.today() - timedelta(days=date.today().weekday()))
    with col_b:
        add_days = st.number_input("Días a generar", 7, 14, 7)
    with col_c:
        if st.button("Generar renglones de semana"):
            rows = []
            for d in range(add_days):
                dd = start + timedelta(days=d)
                for meal in MEAL_OPTS:
                    rows.append({
                        "date": dd.isoformat(),
                        "meal": meal,
                        "item_type": "food",
                        "ref_id": "",
                        "name": "",
                        "servings_or_portions": 0,
                    })
            st.session_state.menu = pd.DataFrame(rows)

    st.session_state.menu = st.data_editor(
        st.session_state.menu,
        num_rows="dynamic",
        height=420,
        column_config={
            "meal": st.column_config.SelectboxColumn(options=MEAL_OPTS),
            "item_type": st.column_config.SelectboxColumn(options=["food", "recipe"]),
        },
    )

    st.download_button(
        "Descargar plan_menu.csv",
        data=st.session_state.menu.to_csv(index=False).encode("utf-8"),
        file_name="plan_menu.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Validación por día vs reglas")
    vv_col1, vv_col2 = st.columns([1, 2])
    with vv_col1:
        days = sorted(set(st.session_state.menu["date"])) if not st.session_state.menu.empty else []
        chosen_day = st.selectbox("Elige día", options=["-"] + days)
        tol = st.slider("Tolerancia %", 0, 30, 10, help="Margen sobre objetivos macro")
    with vv_col2:
        if chosen_day != "-":
            df_day = st.session_state.menu[st.session_state.menu["date"] == chosen_day]
            for meal in MEAL_OPTS:
                items = df_day[df_day["meal"] == meal].to_dict(orient="records")
                if not items:
                    continue
                m = sum_items_macros(items, st.session_state.foods, st.session_state.recipes)
                st.write(f"**{meal}**")
                st.write(pd.DataFrame([m]))
                target = st.session_state.rules.get("macro_targets", {}).get(meal, {})
                if target:
                    msgs = []
                    for k, v in target.items():
                        low, high = v * (1 - tol/100), v * (1 + tol/100)
                        val = m.get(k, 0)
                        if not (low <= val <= high):
                            msgs.append(f"{k}: {val:.0f} fuera de [{low:.0f}, {high:.0f}]")
                    if msgs:
                        st.warning("; ".join(msgs))
                    else:
                        st.success("Dentro de objetivo")

with TAB_SUPER:
    st.subheader("Generar lista de súper")
    if st.session_state.menu.empty:
        st.info("Primero llena el plan semanal en la pestaña anterior.")
    else:
        glist = grocery_from_menu(st.session_state.menu, st.session_state.foods, st.session_state.recipes)
        st.dataframe(glist)
        st.download_button(
            "Descargar lista_super.csv",
            data=glist.to_csv(index=False).encode("utf-8"),
            file_name="lista_super.csv",
            mime="text/csv",
        )

with TAB_CHAT:
    st.subheader("Agente conversacional (sin LLM)")
    st.caption("Escribe por ejemplo: 'Sugiéreme un desayuno sin lácteos' o 'Haz mi lista de súper de la semana'")

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("Escribe tu mensaje…")

    def _assistant_reply(msg: str) -> str:
        text = msg.lower()
        exclusions = st.session_state.rules.get("exclusions", [])
        # Intent 1: lista de súper
        if ("lista" in text) and ("super" in text or "súper" in text):
            if st.session_state.menu.empty:
                return "No tengo un plan cargado. Ve a 'Plan semanal' y agrega tus comidas."
            df = grocery_from_menu(st.session_state.menu, st.session_state.foods, st.session_state.recipes)
            if df.empty:
                return "Tu plan aún no tiene items suficientes para armar la lista."
            # Devolvemos un resumen legible
            lines = ["Tu lista de súper (resumen):"]
            for _, r in df.iterrows():
                lines.append(f"• {r['name']}: {r['qty_compra']} × {r['unit']}")
            return "\n".join(lines)

        # Intent 2: sugerencias por comida
        meal_detected = None
        for mname in MEAL_OPTS:
            if mname in text:
                meal_detected = mname
                break
        if ("sug" in text or "recom" in text) and meal_detected:
            # Una opción combo + algunas recetas
            combo = suggest_combo_for_meal(meal_detected, st.session_state.rules, st.session_state.foods, exclusions)
            recs = suggest_recipes_for_meal(meal_detected, st.session_state.recipes, exclusions)
            lines = [f"Opciones para **{meal_detected}**:"]
            # Combo
            if combo["items"]:
                lines.append("1) Combo por porciones objetivo:")
                for it in combo["items"]:
                    lines.append(f"   - {it['name']} × {it['servings_or_portions']}")
            # Recetas
            if recs:
                lines.append("2) Recetas sugeridas:")
                for r in recs:
                    lines.append(f"   - {r.name} (id {r.recipe_id}) — {r.tags}")
            lines.append("\nTip: En 'Plan semanal' puedes agregar estos items con sus cantidades.")
            return "\n".join(lines)

        return (
            "Puedo: 1) sugerir por comida (di 'sugiéreme [desayuno/comida/cena]'), "
            "2) generar 'lista de súper', y 3) validar macros en 'Plan semanal'."
        )

    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        reply = _assistant_reply(prompt)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

# Fin del archivo
