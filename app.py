from __future__ import annotations

from datetime import date
from io import BytesIO
import re
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Control de proyectos de construcción",
    page_icon="🏗️",
    layout="wide",
)

NO_COLUMN = "— No asignar —"


def demo_data() -> pd.DataFrame:
    """Return a small example dataset with planned and actual project data."""
    today = pd.Timestamp(date.today())

    def day(offset: int) -> str:
        return (today + pd.Timedelta(days=offset)).date().isoformat()

    return pd.DataFrame(
        [
            {
                "Partida": "Movimiento de tierras",
                "Capítulo": "Obras preliminares",
                "Inicio Plan": day(-60),
                "Fin Plan": day(-45),
                "Inicio Real": day(-58),
                "Fin Real": day(-42),
                "Costo Planificado": 18500,
                "Costo Real": 21800,
                "% Avance": 100,
            },
            {
                "Partida": "Cimentaciones",
                "Capítulo": "Estructura",
                "Inicio Plan": day(-44),
                "Fin Plan": day(-18),
                "Inicio Real": day(-40),
                "Fin Real": None,
                "Costo Planificado": 42000,
                "Costo Real": 26700,
                "% Avance": 58,
            },
            {
                "Partida": "Estructura de hormigón",
                "Capítulo": "Estructura",
                "Inicio Plan": day(-17),
                "Fin Plan": day(20),
                "Inicio Real": None,
                "Fin Real": None,
                "Costo Planificado": 68000,
                "Costo Real": None,
                "% Avance": 0,
            },
            {
                "Partida": "Instalación eléctrica",
                "Capítulo": "Instalaciones",
                "Inicio Plan": day(5),
                "Fin Plan": day(45),
                "Inicio Real": None,
                "Fin Real": None,
                "Costo Planificado": 24500,
                "Costo Real": 12100,
                "% Avance": 42,
            },
            {
                "Partida": "Terminaciones",
                "Capítulo": "Acabados",
                "Inicio Plan": day(21),
                "Fin Plan": day(70),
                "Inicio Real": None,
                "Fin Real": None,
                "Costo Planificado": 51000,
                "Costo Real": None,
                "% Avance": 0,
            },
        ]
    )


def read_uploaded_file(uploaded_file, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Read an uploaded workbook sheet or CSV with common text encodings."""
    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix == "xlsx":
        return pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl")
    if suffix == "csv":
        content = uploaded_file.getvalue()
        last_error = None
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(BytesIO(content), encoding=encoding, sep=None, engine="python")
            except (UnicodeDecodeError, pd.errors.ParserError) as error:
                last_error = error
        raise ValueError(f"No se pudo interpretar el CSV: {last_error}")
    raise ValueError("Formato no compatible. Carga un archivo .xlsx o .csv.")


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def find_column(columns, candidates):
    normalized = {normalize_name(column): column for column in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    for column in columns:
        key = normalize_name(column)
        if any(normalize_name(candidate) in key for candidate in candidates):
            return column
    return NO_COLUMN


def parse_number(value):
    """Convert numeric values and common Latin/US formatted strings to floats."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if not text:
        return np.nan
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    text = re.sub(r"[^0-9,.\-]", "", text)
    if not text or text in {"-", ".", ","}:
        return np.nan
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        decimal_or_group = text.rsplit(",", 1)[-1]
        if len(decimal_or_group) == 3 and text.count(",") == 1:
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")
    try:
        number = float(text)
        return -number if negative else number
    except ValueError:
        return np.nan


def parse_percent(value):
    if pd.isna(value):
        return np.nan
    has_percent_sign = "%" in str(value)
    number = parse_number(value)
    if pd.isna(number):
        return np.nan
    if not has_percent_sign and -1 <= number <= 1:
        number *= 100
    return float(np.clip(number, 0, 100))


def safe_ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def date_column(data: pd.DataFrame, column: str) -> tuple[pd.Series, int]:
    if column == NO_COLUMN:
        return pd.Series(pd.NaT, index=data.index), 0
    raw = data[column]
    parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    invalid = int((raw.notna() & parsed.isna()).sum())
    return parsed, invalid


def numeric_column(data: pd.DataFrame, column: str, percentage: bool = False) -> tuple[pd.Series, int]:
    if column == NO_COLUMN:
        return pd.Series(np.nan, index=data.index, dtype=float), 0
    parser = parse_percent if percentage else parse_number
    parsed = data[column].map(parser).astype(float)
    invalid = int((data[column].notna() & parsed.isna()).sum())
    return parsed, invalid


def make_analysis(data: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, dict]:
    """Build normalized fields and per-item schedule, cost, and EVM measures."""
    result = pd.DataFrame(index=data.index)
    result["Partida"] = (
        data[mapping["Partida"]].fillna("Sin nombre").astype(str)
        if mapping["Partida"] != NO_COLUMN
        else "Sin nombre"
    )
    if mapping["Capítulo"] != NO_COLUMN:
        result["Capítulo"] = data[mapping["Capítulo"]].fillna("Sin capítulo").astype(str)
    else:
        result["Capítulo"] = "Sin capítulo"

    invalid_dates = 0
    date_fields = {
        "Inicio plan": "Inicio Plan",
        "Fin plan": "Fin Plan",
        "Inicio real": "Inicio Real",
        "Fin real": "Fin Real",
    }
    for field, role in date_fields.items():
        result[field], invalid = date_column(data, mapping[role])
        invalid_dates += invalid

    invalid_numbers = 0
    result["Costo planificado"], invalid = numeric_column(data, mapping["Costo Planificado"])
    invalid_numbers += invalid
    result["Costo real"], invalid = numeric_column(data, mapping["Costo Real"])
    invalid_numbers += invalid
    result["Avance %"], invalid = numeric_column(data, mapping["% Avance"], percentage=True)
    invalid_numbers += invalid

    missing_costs = int(result["Costo planificado"].isna().sum() + result["Costo real"].isna().sum())
    result["Costo planificado"] = result["Costo planificado"].fillna(0).clip(lower=0)
    result["Costo real"] = result["Costo real"].fillna(0).clip(lower=0)
    result["Avance %"] = result["Avance %"].fillna(0)

    today = pd.Timestamp(date.today())
    plan_start = result["Inicio plan"]
    plan_end = result["Fin plan"]
    actual_start = result["Inicio real"]
    actual_end = result["Fin real"]
    result["Días planificados"] = (plan_end - plan_start).dt.days.clip(lower=0)
    result["Días reales"] = (actual_end - actual_start).dt.days.clip(lower=0)
    effective_end = actual_end.where(actual_end.notna(), today)
    result["Atraso / adelanto (días)"] = np.where(
        plan_end.notna() & actual_start.notna(),
        (effective_end - plan_end).dt.days,
        np.nan,
    )

    scheduled_progress = pd.Series(0.0, index=result.index)
    valid_plan = plan_start.notna() & plan_end.notna()
    plan_duration = (plan_end - plan_start).dt.days
    zero_duration = valid_plan & plan_duration.eq(0)
    scheduled_progress.loc[zero_duration & (today >= plan_end)] = 1.0
    elapsed_progress = valid_plan & plan_duration.gt(0)
    scheduled_progress.loc[elapsed_progress] = (
        (today - plan_start.loc[elapsed_progress]).dt.days
        / plan_duration.loc[elapsed_progress]
    ).clip(0, 1)

    result["PV"] = result["Costo planificado"] * scheduled_progress
    result["EV"] = result["Costo planificado"] * (result["Avance %"] / 100)
    result["AC"] = result["Costo real"]
    result["Variación ($)"] = result["AC"] - result["Costo planificado"]
    result["Pérdida ($)"] = result["Variación ($)"].clip(lower=0)
    result["Desviación (%)"] = result.apply(
        lambda row: safe_ratio(row["Variación ($)"], row["Costo planificado"]) * 100
        if row["Costo planificado"] != 0
        else np.nan,
        axis=1,
    )
    result["CPI"] = result.apply(lambda row: safe_ratio(row["EV"], row["AC"]), axis=1)
    result["SPI"] = result.apply(lambda row: safe_ratio(row["EV"], row["PV"]), axis=1)
    result["EAC"] = result.apply(
        lambda row: (
            row["AC"] + (row["Costo planificado"] - row["EV"]) / row["CPI"]
            if pd.notna(row["CPI"]) and row["CPI"] > 0
            else np.nan
        ),
        axis=1,
    )

    budget = float(result["Costo planificado"].sum())
    actual_cost = float(result["AC"].sum())
    earned_value = float(result["EV"].sum())
    planned_value = float(result["PV"].sum())
    global_cpi = safe_ratio(earned_value, actual_cost)
    global_spi = safe_ratio(earned_value, planned_value)
    global_eac = (
        actual_cost + (budget - earned_value) / global_cpi
        if pd.notna(global_cpi) and global_cpi > 0
        else np.nan
    )
    average_progress = float(result["Avance %"].mean()) if len(result) else 0.0
    global_progress = safe_ratio(earned_value, budget)
    if pd.isna(global_progress):
        global_progress = average_progress / 100

    metrics = {
        "presupuesto": budget,
        "real": actual_cost,
        "pv": planned_value,
        "ev": earned_value,
        "ac": actual_cost,
        "cpi": global_cpi,
        "spi": global_spi,
        "eac": global_eac,
        "avance": global_progress * 100,
        "perdidas": float(result["Pérdida ($)"].sum()),
        "atraso": float(result["Atraso / adelanto (días)"].clip(lower=0).sum(skipna=True)),
        "invalid_dates": invalid_dates,
        "invalid_numbers": invalid_numbers,
        "missing_costs": missing_costs,
    }
    return result, metrics


def money(value):
    if pd.isna(value):
        return "N/D"
    return f"${value:,.2f}"


def render_costs(analysis: pd.DataFrame):
    st.subheader("Partidas con mayores pérdidas")
    losses = analysis.nlargest(10, "Pérdida ($)").sort_values("Pérdida ($)")
    if losses.empty:
        st.info("No hay partidas para mostrar.")
    else:
        figure = px.bar(
            losses,
            x="Pérdida ($)",
            y="Partida",
            orientation="h",
            color="Pérdida ($)",
            color_continuous_scale=["#f7b7b2", "#b42318"],
            labels={"Pérdida ($)": "Pérdida", "Partida": ""},
        )
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Costo planificado vs. real por capítulo")
    chapter_costs = (
        analysis.groupby("Capítulo", dropna=False)[["Costo planificado", "Costo real"]]
        .sum()
        .reset_index()
        .melt(id_vars="Capítulo", var_name="Tipo", value_name="Costo")
    )
    figure = px.bar(
        chapter_costs,
        x="Capítulo",
        y="Costo",
        color="Tipo",
        barmode="group",
        labels={"Costo": "Costo", "Capítulo": ""},
        color_discrete_map={"Costo planificado": "#2563eb", "Costo real": "#f97316"},
    )
    st.plotly_chart(figure, use_container_width=True)


def render_schedule(analysis: pd.DataFrame):
    st.subheader("Cronograma planificado vs. real")
    plan = analysis.loc[
        analysis["Inicio plan"].notna() & analysis["Fin plan"].notna(),
        ["Partida", "Inicio plan", "Fin plan"],
    ].copy()
    plan.columns = ["Partida", "Inicio", "Fin"]
    plan["Serie"] = "Planificado"

    actual = analysis.loc[analysis["Inicio real"].notna(), ["Partida", "Inicio real", "Fin real"]].copy()
    actual["Fin real"] = actual["Fin real"].fillna(pd.Timestamp(date.today()))
    actual.columns = ["Partida", "Inicio", "Fin"]
    actual["Serie"] = "Real"
    gantt_data = pd.concat([plan, actual], ignore_index=True)
    gantt_data = gantt_data.loc[gantt_data["Fin"] >= gantt_data["Inicio"]]
    if gantt_data.empty:
        st.info("Asigna fechas planificadas o reales para visualizar el cronograma.")
    else:
        gantt_data["Tarea"] = gantt_data["Partida"] + " · " + gantt_data["Serie"]
        figure = px.timeline(
            gantt_data,
            x_start="Inicio",
            x_end="Fin",
            y="Tarea",
            color="Serie",
            color_discrete_map={"Planificado": "#2563eb", "Real": "#f97316"},
            labels={"Tarea": "", "Serie": ""},
        )
        figure.update_yaxes(autorange="reversed")
        figure.update_layout(height=max(420, min(900, 28 * len(gantt_data))))
        st.plotly_chart(figure, use_container_width=True)

    late = analysis.loc[
        analysis["Atraso / adelanto (días)"].notna()
        & analysis["Atraso / adelanto (días)"].ne(0),
        ["Partida", "Inicio plan", "Fin plan", "Inicio real", "Fin real", "Atraso / adelanto (días)"],
    ].copy()
    late["Estado"] = np.where(late["Atraso / adelanto (días)"] > 0, "Atraso", "Adelanto")
    late["Atraso / adelanto (días)"] = late["Atraso / adelanto (días)"].astype(int)
    st.subheader("Atrasos y adelantos")
    if late.empty:
        st.success("No se registran atrasos ni adelantos en las partidas con inicio real.")
    else:
        st.dataframe(late, use_container_width=True, hide_index=True)


def style_analysis(data: pd.DataFrame):
    def highlight_negative_outcomes(series):
        styles = []
        for value in series:
            styles.append(
                "background-color: #fee2e2; color: #991b1b; font-weight: 600"
                if pd.notna(value) and value > 0
                else ""
            )
        return styles

    styled = data.style
    for column in ("Pérdida ($)", "Atraso / adelanto (días)"):
        if column in data.columns:
            styled = styled.apply(highlight_negative_outcomes, subset=[column])
    return styled.format(
        {
            "Costo planificado": "${:,.2f}",
            "Costo real": "${:,.2f}",
            "Variación ($)": "${:,.2f}",
            "Pérdida ($)": "${:,.2f}",
            "PV": "${:,.2f}",
            "EV": "${:,.2f}",
            "AC": "${:,.2f}",
            "EAC": "${:,.2f}",
            "Desviación (%)": "{:,.1f}%",
            "Avance %": "{:,.1f}%",
            "CPI": "{:,.2f}",
            "SPI": "{:,.2f}",
            "Días planificados": "{:,.0f}",
            "Días reales": "{:,.0f}",
            "Atraso / adelanto (días)": "{:,.0f}",
        },
        na_rep="N/D",
    )


def main():
    st.title("🏗️ Control de proyectos de construcción")
    st.caption("Análisis de costos, plazos y Valor Ganado (EVM)")

    with st.sidebar:
        st.header("Carga de datos")
        if st.button("Datos Demo", use_container_width=True):
            st.session_state["use_demo"] = True
        uploaded_file = st.file_uploader("Carga un archivo .xlsx o .csv", type=["xlsx", "csv"])
        if uploaded_file is not None and st.button("Usar archivo cargado", use_container_width=True):
            st.session_state["use_demo"] = False

    use_demo = st.session_state.get("use_demo", False)
    if uploaded_file is None and not use_demo:
        st.info("Carga el presupuesto Excel/CSV o selecciona **Datos Demo** en la barra lateral.")
        return

    try:
        if use_demo:
            raw_data = demo_data()
            st.sidebar.caption("Fuente activa: Datos Demo")
        else:
            sheet_name = None
            if uploaded_file.name.lower().endswith(".xlsx"):
                workbook = pd.ExcelFile(uploaded_file, engine="openpyxl")
                if not workbook.sheet_names:
                    st.error("El libro de Excel no contiene hojas.")
                    return
                sheet_name = st.sidebar.selectbox("Hoja de Excel", workbook.sheet_names)
                uploaded_file.seek(0)
            raw_data = read_uploaded_file(uploaded_file, uploaded_file.name, sheet_name)
            st.sidebar.caption(f"Fuente activa: {uploaded_file.name}")
    except Exception as error:
        st.error(f"No se pudo cargar el archivo: {error}")
        return

    if raw_data.empty or len(raw_data.columns) == 0:
        st.error("El archivo no contiene datos para analizar.")
        return

    columns = list(raw_data.columns)
    aliases = {
        "Partida": ["Partida", "Actividad", "Descripción", "Item", "Concepto", "Tarea"],
        "Capítulo": ["Capítulo", "Capitulo", "Chapter"],
        "Inicio Plan": ["Inicio Plan", "Fecha Inicio Plan", "Inicio Programado", "Fecha Inicio Programada"],
        "Fin Plan": ["Fin Plan", "Fecha Fin Plan", "Fin Programado", "Fecha Fin Programada", "Término Plan"],
        "Inicio Real": ["Inicio Real", "Fecha Inicio Real", "Inicio Ejecutado"],
        "Fin Real": ["Fin Real", "Fecha Fin Real", "Fin Ejecutado", "Término Real"],
        "Costo Planificado": ["Costo Planificado", "Costo Plan", "Presupuesto", "Monto Planificado", "Planificado"],
        "Costo Real": ["Costo Real", "Costo Ejecutado", "Gasto Real"],
        "% Avance": ["% Avance", "Avance", "Porcentaje Avance", "Progreso", "% Completado"],
    }
    defaults = {
        role: find_column(columns, candidates) for role, candidates in aliases.items()
    }
    if defaults["Partida"] == NO_COLUMN:
        defaults["Partida"] = columns[0]

    with st.sidebar:
        st.header("Mapeo de columnas")
        mapping = {}
        role_order = [
            "Partida",
            "Capítulo",
            "Inicio Plan",
            "Fin Plan",
            "Inicio Real",
            "Fin Real",
            "Costo Planificado",
            "Costo Real",
            "% Avance",
        ]
        for role in role_order:
            options = [NO_COLUMN] + columns
            default = defaults[role]
            index = options.index(default) if default in options else 0
            mapping[role] = st.selectbox(
                role,
                options,
                index=index,
                key=f"column_{role}",
            )

    if mapping["Partida"] == NO_COLUMN:
        st.error("Asigna una columna para Partida.")
        return

    analysis, metrics = make_analysis(raw_data, mapping)
    if metrics["invalid_dates"]:
        st.warning(
            f"Se ignoraron {metrics['invalid_dates']} valores de fecha inválidos o no interpretables."
        )
    if metrics["invalid_numbers"]:
        st.warning(
            f"Se encontraron {metrics['invalid_numbers']} valores de costo/avance no numéricos; "
            "se trataron como nulos."
        )
    if metrics["missing_costs"]:
        st.info(
            f"{metrics['missing_costs']} valores de costo están vacíos; se consideran $0 para los totales."
        )

    with st.expander("Vista previa del archivo cargado"):
        st.dataframe(raw_data.head(20), use_container_width=True, hide_index=True)

    kpi_tab, cost_tab, schedule_tab, data_tab = st.tabs(
        ["KPIs", "Costos", "Cronograma", "Datos"]
    )
    with kpi_tab:
        first_row = st.columns(5)
        first_row[0].metric("Pérdidas totales", money(metrics["perdidas"]))
        first_row[1].metric("Días de atraso", f'{metrics["atraso"]:,.0f}')
        first_row[2].metric("CPI", f'{metrics["cpi"]:.2f}' if pd.notna(metrics["cpi"]) else "N/D")
        first_row[3].metric("SPI", f'{metrics["spi"]:.2f}' if pd.notna(metrics["spi"]) else "N/D")
        first_row[4].metric("Avance global", f'{metrics["avance"]:.1f}%')
        st.divider()
        second_row = st.columns(4)
        second_row[0].metric("Presupuesto (BAC)", money(metrics["presupuesto"]))
        second_row[1].metric("Costo real (AC)", money(metrics["ac"]))
        second_row[2].metric("Valor planificado (PV)", money(metrics["pv"]))
        second_row[3].metric("Valor ganado (EV)", money(metrics["ev"]))
        st.metric("Proyección de costo final (EAC)", money(metrics["eac"]))
        st.caption(
            "PV se estima con el avance temporal entre Inicio/Fin Plan; EV = presupuesto × avance real. "
            "EAC = AC + (BAC − EV) / CPI."
        )

    with cost_tab:
        render_costs(analysis)

    with schedule_tab:
        render_schedule(analysis)

    with data_tab:
        display_columns = [
            "Partida",
            "Capítulo",
            "Inicio plan",
            "Fin plan",
            "Inicio real",
            "Fin real",
            "Días planificados",
            "Días reales",
            "Atraso / adelanto (días)",
            "Costo planificado",
            "Costo real",
            "Variación ($)",
            "Pérdida ($)",
            "Desviación (%)",
            "Avance %",
            "PV",
            "EV",
            "AC",
            "CPI",
            "SPI",
            "EAC",
        ]
        st.dataframe(
            style_analysis(analysis[display_columns]),
            use_container_width=True,
            hide_index=True,
        )
        output = BytesIO()
        try:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                analysis[display_columns].to_excel(writer, index=False, sheet_name="Análisis")
                raw_data.to_excel(writer, index=False, sheet_name="Datos originales")
            st.download_button(
                "Descargar análisis en Excel",
                data=output.getvalue(),
                file_name="analisis_proyecto_construccion.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as error:
            st.error(f"No se pudo generar el Excel de descarga: {error}")


if __name__ == "__main__":
    main()
