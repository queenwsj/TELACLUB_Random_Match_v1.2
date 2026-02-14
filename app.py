import streamlit as st
import pandas as pd
import random
import math
import io

# ==========================================
# 1. 로직 엔진 (기존 로직 100% 동일)
# ==========================================
def base_name(p: str) -> str:
    return p.split("(")[0].strip()

def get_gender(code: str) -> str:
    c = base_name(code)
    if len(c) >= 2:
        if c[1] == "M": return "M"
        if c[1] == "W": return "W"
    return "U"

def match_label(players4):
    genders = [get_gender(p) for p in players4]
    m = genders.count("M")
    w = genders.count("W")
    if m == 4: return "남복Match"
    if w == 4: return "여복Match"
    if m == 2 and w == 2: return "혼복Match"
    if m == 3 and w == 1: return "잡복(남3여1)"
    if m == 1 and w == 3: return "잡복(여3남1)"
    return "랜덤매칭"

def team_key(t): return frozenset(map(base_name, t))
def opp_key(a, b): return frozenset([base_name(a), base_name(b)])
def mixed_partner_keys_in_team(t):
    p1, p2 = t
    g1, g2 = get_gender(p1), get_gender(p2)
    if (g1, g2) in [("M", "W"), ("W", "M")]:
        return {frozenset([base_name(p1), base_name(p2)])}
    return set()

def commit_match(t1, t2, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used):
    tk1 = team_key(t1); tk2 = team_key(t2)
    teammate_used.add(tk1); teammate_used.add(tk2)
    round_teammate_used.add(tk1); round_teammate_used.add(tk2)
    for x in t1:
        for y in t2:
            ok = opp_key(x, y)
            opponent_used.add(ok); round_opponent_used.add(ok)
    for k in mixed_partner_keys_in_team(t1):
        mixed_partner_used.add(k); round_mixed_partner_used.add(k)
    for k in mixed_partner_keys_in_team(t2):
        mixed_partner_used.add(k); round_mixed_partner_used.add(k)

def best_pairing_of_four(players4, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used):
    a, b, c, d = players4
    all_pairings = [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
    genders = [get_gender(p) for p in players4]
    m, w = genders.count("M"), genders.count("W")
    
    def is_mixed_team(t):
        gs = [get_gender(x) for x in t]
        return gs.count("M") == 1 and gs.count("W") == 1

    if m == 2 and w == 2:
        pairings = [(t1, t2) for (t1, t2) in all_pairings if is_mixed_team(t1) and is_mixed_team(t2)]
        if not pairings: pairings = all_pairings
    else:
        pairings = all_pairings

    def score(t1, t2):
        pen_team_r = (1 if team_key(t1) in round_teammate_used else 0) + (1 if team_key(t2) in round_teammate_used else 0)
        pen_team_t = (1 if team_key(t1) in teammate_used else 0) + (1 if team_key(t2) in teammate_used else 0)
        pen_opp_r, pen_opp_t = 0, 0
        for x in t1:
            for y in t2:
                ok = opp_key(x, y)
                if ok in round_opponent_used: pen_opp_r += 1
                if ok in opponent_used: pen_opp_t += 1
        pen_mix_r, pen_mix_t = 0, 0
        for k in mixed_partner_keys_in_team(t1) | mixed_partner_keys_in_team(t2):
            if k in round_mixed_partner_used: pen_mix_r += 1
            if k in mixed_partner_used: pen_mix_t += 1
        return pen_team_r * 5000 + pen_team_t * 1000 + pen_mix_r * 500 + pen_mix_t * 100 + pen_opp_r * 50 + pen_opp_t * 10

    best, best_s = None, float('inf')
    random.shuffle(pairings)
    for t1, t2 in pairings:
        s = score(t1, t2)
        if s < best_s: best_s = s; best = (t1, t2)
    return best

def build_groups_by_priority(pool):
    men = [p for p in pool if get_gender(p) == "M"]
    women = [p for p in pool if get_gender(p) == "W"]
    unknown = [p for p in pool if get_gender(p) == "U"]
    random.shuffle(men); random.shuffle(women); random.shuffle(unknown)
    groups = []
    while len(men) >= 4: groups.append([men.pop() for _ in range(4)])
    while len(women) >= 4: groups.append([women.pop() for _ in range(4)])
    while len(men) >= 2 and len(women) >= 2: groups.append([men.pop(), men.pop(), women.pop(), women.pop()])
    while len(men) >= 3 and len(women) >= 1: groups.append([men.pop(), men.pop(), men.pop(), women.pop()])
    while len(women) >= 3 and len(men) >= 1: groups.append([women.pop(), women.pop(), women.pop(), men.pop()])
    leftovers = men + women + unknown
    return groups, leftovers

def make_matches(pool, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used):
    matches = []
    groups, leftovers = build_groups_by_priority(pool)
    if len(leftovers) >= 4:
        while len(leftovers) >= 4: groups.append([leftovers.pop() for _ in range(4)])

    for g in groups:
        random.shuffle(g)
        t1, t2 = best_pairing_of_four(g, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used)
        commit_match(t1, t2, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used)
        matches.append({"team1": t1, "team2": t2, "type": match_label(g)})
    return matches, leftovers

def build_event_games(players, usage, min_games=3, max_games=4):
    add = {}; room = {}
    for p in players:
        current = usage.get(p, 0)
        room[p] = max(0, max_games - current)
        need = max(0, min_games - current)
        add[p] = min(need, room[p])
    
    total_slots = sum(add.values())
    if total_slots == 0: return []

    def pick_filler():
        cands1 = [p for p in players if add[p] == 0 and room[p] > add[p]]
        if cands1: cands1.sort(key=lambda x: (usage[x], random.random())); return cands1[0]
        cands2 = [p for p in players if room[p] > add[p]]
        if cands2: cands2.sort(key=lambda x: (usage[x], random.random())); return cands2[0]
        return None

    while sum(1 for p in players if add[p] > 0) < 4:
        p = pick_filler()
        if not p: break
        add[p] += 1; total_slots += 1

    while total_slots % 4 != 0:
        p = pick_filler()
        if not p: break
        add[p] += 1; total_slots += 1

    games = []
    remaining = add.copy()
    while True:
        active = [p for p in players if remaining[p] > 0]
        if not active or len(active) < 4: break
        active.sort(key=lambda x: (-remaining[x], usage[x], random.random()))
        chosen = active[:4]
        for p in chosen: remaining[p] -= 1
        games.append(chosen)
    return games

def decorate_event_games(games):
    seen = {}
    decorated_games = []
    for g in games:
        dg = []
        for p in g:
            seen[p] = seen.get(p, 0) + 1
            if seen[p] >= 2: dg.append(p + "(중복)")
            else: dg.append(p)
        random.shuffle(dg)
        decorated_games.append(dg)
    return decorated_games

def generate_schedule(am, aw, bm, bw):
    leagues = {
        "A": [f"AM{i+1:02d}" for i in range(am)] + [f"AW{i+1:02d}" for i in range(aw)],
        "B": [f"BM{i+1:02d}" for i in range(bm)] + [f"BW{i+1:02d}" for i in range(bw)],
    }
    results = []
    
    for league_name, players in leagues.items():
        if not players: continue
        usage = {p: 0 for p in players}
        teammate_used = set(); opponent_used = set(); mixed_partner_used = set()

        for r in range(1, 4):
            round_teammate_used = set(); round_opponent_used = set(); round_mixed_partner_used = set()
            order = players[:]
            random.shuffle(order)
            order.sort(key=lambda x: usage.get(x, 0))
            
            capacity = (len(order) // 4) * 4
            participants = order[:capacity]
            
            round_matches, _ = make_matches(participants, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used)
            
            for m in round_matches:
                t1, t2 = m["team1"], m["team2"]
                for p in (t1+t2): usage[base_name(p)] += 1
                results.append({"round": f"{r}R", "league": f"{league_name}리그", "team1": t1, "team2": t2, "note": m["type"]})

        games = build_event_games(players, usage)
        games = decorate_event_games(games)
        if games:
            round_teammate_used = set(); round_opponent_used = set(); round_mixed_partner_used = set()
            for g in games:
                t1, t2 = best_pairing_of_four(g, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used)
                commit_match(t1, t2, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used)
                for p in (t1+t2): usage[base_name(p)] += 1
                note = match_label(g)
                if any("(중복)" in x for x in (t1+t2)): note += " (중복)"
                results.append({"round": "4R (이벤트)", "league": f"{league_name}리그", "team1": t1, "team2": t2, "note": note})
                
    return results

def calculate_stats(schedule_data):
    stats = {}
    for row in schedule_data:
        r_num = row["round"]
        league = row["league"]
        t1, t2 = row["team1"], row["team2"]
        for p_raw in (t1 + t2):
            p_name = base_name(p_raw)
            if p_name not in stats:
                stats[p_name] = {"League": league, "1R": 0, "2R": 0, "3R": 0, "4R": 0, "Total": 0}
            if "1R" in r_num: stats[p_name]["1R"] += 1
            elif "2R" in r_num: stats[p_name]["2R"] += 1
            elif "3R" in r_num: stats[p_name]["3R"] += 1
            elif "4R" in r_num: stats[p_name]["4R"] += 1
            stats[p_name]["Total"] += 1
    
    data = []
    for name, info in stats.items():
        data.append({
            "리그": info["League"], "이름": name, 
            "1R": info["1R"], "2R": info["2R"], "3R": info["3R"], "4R": info["4R"], "총합": info["Total"]
        })
    return pd.DataFrame(data).sort_values(by=["리그", "이름"])

# ==========================================
# 2. 스트림릿(Streamlit) 화면 구성
# ==========================================
st.set_page_config(page_title="TELA Tennis Match", page_icon="🎾", layout="wide")

st.title("🎾 TELA CLUB Random Match")
st.markdown("모바일/PC 어디서든 사용 가능한 랜덤 매치 생성기입니다. (3경기 보장 / 4경기 제한)")

# 사이드바 입력
st.sidebar.header("참가 인원 설정")
col1, col2 = st.sidebar.columns(2)
with col1:
    am = st.number_input("A리그(남)", min_value=0, value=8)
    aw = st.number_input("A리그(여)", min_value=0, value=1)
with col2:
    bm = st.number_input("B리그(남)", min_value=0, value=3)
    bw = st.number_input("B리그(여)", min_value=0, value=2)

if st.sidebar.button("대진표 생성", type="primary"):
    data = generate_schedule(am, aw, bm, bw)
    
    # 데이터프레임 변환 (화면 표시용)
    # [수정된 부분] 팀1, 팀2를 각각 2개의 칸으로 분리
    display_data = []
    for d in data:
        display_data.append({
            "라운드": d["round"],
            "리그": d["league"],
            "팀1 (1)": d['team1'][0],
            "팀1 (2)": d['team1'][1],
            "팀2 (1)": d['team2'][0],
            "팀2 (2)": d['team2'][1],
            "비고": d["note"]
        })
    df_matches = pd.DataFrame(display_data)

    # 1. 대진표 탭
    tab1, tab2 = st.tabs(["📋 대진표", "📊 출전 현황"])
    
    with tab1:
        st.subheader("경기 매치업")
        
        # 라운드별 스타일링을 위한 함수
        def highlight_rows(row):
            if "A리그" in row["리그"]:
                return ['background-color: #C8E6C9; color: black'] * len(row)
            else:
                return ['background-color: #BBDEFB; color: black'] * len(row)

        st.dataframe(df_matches.style.apply(highlight_rows, axis=1), use_container_width=True, height=600)

    # 2. 출전 현황 탭
    with tab2:
        st.subheader("선수별 출전 기록")
        df_stats = calculate_stats(data)
        
        def highlight_stats(val):
            if isinstance(val, int):
                if val < 3: return 'background-color: #FFCDD2; color: black' # 빨강
                if val == 3: return 'background-color: #E8F5E9; color: black' # 초록
                if val >= 4: return 'background-color: #FFF9C4; color: black' # 노랑
            return ''

        st.dataframe(df_stats.style.applymap(highlight_stats, subset=["총합"]), use_container_width=True)

    # 엑셀 다운로드 버튼
    # 메모리 내에서 엑셀 파일 생성
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_matches.to_excel(writer, sheet_name='대진표', index=False)
        df_stats.to_excel(writer, sheet_name='출전현황', index=False)
    
    st.sidebar.download_button(
        label="📥 엑셀 파일 다운로드",
        data=output.getvalue(),
        file_name="TELA_대진표_결과.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("왼쪽 사이드바에서 인원을 설정하고 '대진표 생성' 버튼을 눌러주세요.")
