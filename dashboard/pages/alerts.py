"""Alert queue page."""

import streamlit as st

from dashboard.cases import load_case_queue, render_case, synchronize_cases


def render() -> None:
    st.markdown('<div class="finshield-kicker">ALERT OPERATIONS</div>', unsafe_allow_html=True)
    st.title("Prioritized alert queue")
    queue = load_case_queue()
    cases = synchronize_cases(queue)
    queue["Workflow"] = queue["Case"].map(lambda case_id: cases[case_id].status.value)
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        status = st.segmented_control("Status", ["All", "Critical", "Review", "Monitor", "Low risk"], default="All")
        filtered = queue if status == "All" else queue.loc[queue["Status"] == status]
        event = st.dataframe(
            filtered[["Case", "Transaction", "Amount", "Risk", "Status", "Workflow"]],
            use_container_width=True, hide_index=True, on_select="rerun",
            selection_mode="single-row", key="case_queue",
            column_config={
                "Amount": st.column_config.NumberColumn(format="₹ %.2f"),
                "Risk": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            },
        )
        selected_rows = event.selection.rows
        if selected_rows:
            st.session_state["selected_case"] = filtered.iloc[selected_rows[0]]["Case"]
        st.caption("Select any row to open the investigation panel. Queue records are scored labelled PaySim examples.")
    with right:
        case_id = st.session_state.get("selected_case")
        if case_id and case_id in set(queue["Case"]):
            render_case(queue.loc[queue["Case"] == case_id].iloc[0])
            if st.button("Open full investigation", use_container_width=True, type="primary"):
                st.session_state["open_investigation"] = True
                st.rerun()
        else:
            st.info("Select a case to inspect its risk evidence.")

    if st.session_state.get("open_investigation") and st.session_state.get("selected_case"):
        st.divider()
        st.markdown("## Investigation activity")
        case = queue.loc[queue["Case"] == st.session_state["selected_case"]].iloc[0]
        render_case(case)
        notes = st.text_area("Investigator notes", key=f"notes_{case['Case']}", placeholder="Document evidence and reasoning…")
        disposition = st.selectbox("Disposition", ["IN_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "INCONCLUSIVE"], key=f"disposition_{case['Case']}")
        if st.button("Save case update", key=f"save_{case['Case']}"):
            st.session_state[f"saved_{case['Case']}"] = {"notes": notes, "disposition": disposition}
            st.success(f"{case['Case']} updated for this dashboard session.")
