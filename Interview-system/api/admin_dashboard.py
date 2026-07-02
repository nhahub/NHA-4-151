"""
Admin Dashboard — Monitor ongoing and completed AI interviews.
"""

import streamlit as st
import sqlite3
import json
from datetime import datetime

st.set_page_config(page_title="Interview Admin Dashboard", layout="wide", page_icon="🎙️")

st.title("🎙️ AI Voice Interview System — Admin Dashboard")
st.markdown("Monitor ongoing and completed interviews stored in the SQLite checkpoint database.")

DB_PATH = "checkpoints.db"

def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        st.error(f"Could not connect to database: {e}")
        return None

def fetch_sessions():
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        # Checkpoints table typically stores: thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
        cursor = conn.cursor()
        
        # Get latest checkpoint for each thread_id
        cursor.execute('''
            SELECT thread_id, checkpoint
            FROM checkpoints
            WHERE checkpoint_id IN (
                SELECT MAX(checkpoint_id)
                FROM checkpoints
                GROUP BY thread_id
            )
            ORDER BY checkpoint_id DESC
        ''')
        
        rows = cursor.fetchall()
        sessions = []
        for row in rows:
            thread_id = row['thread_id']
            try:
                # The checkpoint data might be pickled or JSON.
                # In langgraph-checkpoint-sqlite, checkpoint is a BLOB of pickled data if using pickle, or JSON if using JSON serializer.
                # Actually, langgraph default serializer is jsonplus. We can try to decode it.
                raw_cp = row['checkpoint']
                # Try simple json decode (might have some jsonplus structure)
                # But it's usually bytes.
                import pickle
                try:
                    cp_dict = pickle.loads(raw_cp)
                except:
                    # If it's JSON bytes
                    cp_dict = json.loads(raw_cp.decode('utf-8'))
                
                # State is inside cp_dict['channel_values']
                state = cp_dict.get('channel_values', {})
                if not state:
                     continue
                     
                sessions.append({
                    "session_id": thread_id,
                    "role_title": state.get("role_title", "Unknown Role"),
                    "seniority": state.get("seniority", "Unknown"),
                    "turn": state.get("turn", 0),
                    "phase": state.get("phase", "Unknown"),
                    "report": state.get("report", None)
                })
            except Exception as e:
                pass
                
        return sessions
    except Exception as e:
        # Table might not exist yet
        st.warning(f"Database error (no sessions yet?): {e}")
        return []
    finally:
        conn.close()

sessions = fetch_sessions()

if not sessions:
    st.info("No interview sessions found in the database.")
else:
    # Top metrics
    completed = [s for s in sessions if s.get("report") is not None]
    ongoing = [s for s in sessions if s.get("report") is None]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sessions", len(sessions))
    col2.metric("Ongoing Interviews", len(ongoing))
    col3.metric("Completed Interviews", len(completed))
    
    st.markdown("---")
    
    # Session list
    st.subheader("Interview Sessions")
    
    # We will show a dataframe for overview, and expanders for details
    for s in sessions:
        status = "✅ Completed" if s["report"] else "🔄 In Progress"
        score = ""
        if s["report"]:
             score = f" - Score: {s['report'].get('overall_score', 0):.1f}/10 ({s['report'].get('recommendation', '')})"
             
        with st.expander(f"{status} | {s['role_title']} ({s['seniority']}) | Session: {s['session_id'][:8]}... | Turn: {s['turn']} {score}"):
            st.write(f"**Session ID:** `{s['session_id']}`")
            st.write(f"**Current Phase:** {s['phase']}")
            
            if s["report"]:
                st.success(f"**Recommendation:** {s['report'].get('recommendation')}")
                st.json(s["report"])
            else:
                st.info("Interview currently in progress...")
                
if st.button("Refresh Data"):
    st.rerun()
