import streamlit as st
import pandas as pd
import random
import math
import io

# ==========================================
# 1. 로직 엔진
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

def build_groups_by_priority(pool, mixed_usage, max_groups=None):
    if max_groups is None:
        max_groups = len(pool) // 4
        
    groups = []
    pool_copy = pool.copy()

    while len(groups) < max_groups and len(pool_copy) >= 4:
        # 1. 출전이 가장 시급한 선수를 기둥(Anchor)으로 세웁니다.
        anchor = pool_copy.pop(0)
        g_anchor = get_gender(anchor)
        
        group = [anchor]
        
        men = [p for p in pool_copy if get_gender(p) == "M"]
        women = [p for p in pool_copy if get_gender(p) == "W"]
        
        def extract_players(p_list, count, is_mixed_target):
            if is_mixed_target:
                # 혼성 경기면 혼복 경험 적은 순 우선
                p_list.sort(key=lambda x: (mixed_usage.get(base_name(x), 0), pool.index(x)))
            else:
                # 동성 경기면 순수하게 전체 출전 적은 순 우선
                p_list.sort(key=lambda x: pool.index(x))
                
            extracted = []
            for _ in range(count):
                if p_list:
                    p = p_list.pop(0)
                    extracted.append(p)
                    if p in pool_copy:
                        pool_copy.remove(p)
            return extracted

        # 2. 무조건 [동성복 > 혼복 > 잡복] 순으로 엄격하게 우선순위 적용
        choice = None
        if g_anchor == "M":
            if len(men) >= 3:
                choice = "M4"  # 1순위: 남복
            elif len(men) >= 1 and len(women) >= 2:
                choice = "M2W2" # 2순위: 혼복
        elif g_anchor == "W":
            if len(women) >= 3:
                choice = "W4"  # 1순위: 여복
            elif len(women) >= 1 and len(men) >= 2:
                choice = "M2W2" # 2순위: 혼복

        # 3. 결정된 매치 타입에 맞춰 선수 투입
        if choice == "M4":
            group.extend(extract_players(men, 3, False))
        elif choice == "W4":
            group.extend(extract_players(women, 3, False))
        elif choice == "M2W2":
            if g_anchor == "M":
                group.extend(extract_players(men, 1, True))
                group.extend(extract_players(women, 2, True))
            else:
                group.extend(extract_players(women, 1, True))
                group.extend(extract_players(men, 2, True))
        else:
            # 3순위: 동성복/혼복 모두 수학적으로 불가능할 때만 어쩔 수 없이 잡복(Fallback) 구성
            if g_anchor == "M" and len(women) >= 3:
                group.extend(extract_players(women, 3, True))
            elif g_anchor == "W" and len(men) >= 3:
                group.extend(extract_players(men, 3, True))
            else:
                group.extend(extract_players(pool_copy, 3, True))

        groups.append(group)
        
    leftovers = pool_copy
    return groups, leftovers

def make_matches(pool, mixed_usage, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used):
    matches = []
    groups, leftovers = build_groups_by_priority(pool, mixed_usage)
    
    for g in groups:
        random.shuffle(g)
        t1, t2 = best_pairing_of_four(g, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used)
        commit_match(t1, t2, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used)
        matches.append({"team1": t1, "team2": t2, "type": match_label(g)})
    return matches, leftovers

def build_event_games(players, usage, mixed_usage, min_games=3, max_games=4):
    add = {}; room = {}
    for p in players:
        current = usage.get(p, 0)
        room[p] = max(0, max_games - current)
        need = max(0, min_games - current)
        add[p] = min(need, room[p])
    
    total_slots = sum(add.values())
    if total_slots == 0: return []

    def pick_filler():
        current_m = sum(v for p, v in add.items() if get_gender(p) == "M")
        current_w = sum(v for p, v in add.items() if get_gender(p) == "W")
        pref = None
        if current_m % 2 != 0: pref = "M"
        elif current_w % 2 != 0: pref = "W"

        cands1 = [p for p in players if add[p] == 0 and room[p] > add[p]]
        if cands1:
            cands1.sort(key=lambda x: (0 if get_gender(x) == pref else 1, usage[x], random.random()))
            return cands1[0]
            
        cands2 = [p for p in players if room[p] > add[p]]
        if cands2:
            cands2.sort(key=lambda x: (0 if get_gender(x) == pref else 1, usage[x], random.random()))
            return cands2[0]
            
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
        
        groups, _ = build_groups_by_priority(active, mixed_usage, max_groups=1)
        if groups:
            chosen = groups[0]
        else:
            chosen = active[:4]
            
        for p in chosen: remaining[p] -= 1
        games.append(chosen)
    return games

def decorate_event_games(games, usage, min_games=3):
    current_usage = usage.copy()
    decorated_games = []
    
    for g in games:
        dg = []
        for p in g:
            current_usage[p] = current_usage.get(p, 0) + 1
            if current_usage[p] > min_games:
                dg.append(p + "(중복)")
            else:
                dg.append(p)
                
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
        mixed_usage = {p: 0 for p in players}
        teammate_used = set(); opponent_used = set(); mixed_partner_used = set()

        for r in range(1, 4):
            round_teammate_used = set(); round_opponent_used = set(); round_mixed_partner_used = set()
            order = players[:]
            random.shuffle(order)
            order.sort(key=lambda x: usage.get(x, 0))
            
            round_matches, _ = make_matches(order, mixed_usage, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used)
            
            for m in round_matches:
                t1, t2 = m["team1"], m["team2"]
                group = list(t1) + list(t2)
                
                is_mixed = False
                genders = [get_gender(x) for x in group]
                if "M" in genders and "W" in genders:
                    is_mixed = True
                    
                for p in group:
                    b_name = base_name(p)
                    usage[b_name] += 1
                    if is_mixed:
                        mixed_usage[b_name] += 1
                        
                results.append({"round": f"{r}R", "league": f"{league_name}리그", "team1": t1, "team2": t2, "note": m["type"]})

        games = build_event_games(players, usage, mixed_usage)
        games = decorate_event_games(games, usage)
        if games:
            round_teammate_used = set(); round_opponent_used = set(); round_mixed_partner_used = set()
            for g in games:
                t1, t2 = best_pairing_of_four(g, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used)
                commit_match(t1, t2, teammate_used, opponent_used, mixed_partner_used, round_teammate_used, round_opponent_used, round_mixed_partner_used)
                
                group = list(t1) + list(t2)
                is_mixed = False
                genders = [get_gender(x) for x in group]
                if "M" in genders and "W" in genders:
                    is_mixed = True
                    
                for p in group:
                    b_name = base_name(p)
                    usage[b_name] += 1
                    if is_mixed:
                        mixed_usage[b_name] += 1
                        
                note = match_label(g)
                if any("(중복)" in x for x in group): note += " (중복)"
                results.append({"round": "4R (이벤트)", "league": f"{league_name}리그", "team1": t1, "team2": t2, "note": note})
                
    return results

def calculate_stats(schedule_data):
    stats = {}
    for row in schedule_data:
        r_num = row["round"]
        league = row["league"]
        t1, t2 = row["team1"], row["team2"]
        
        match_type = row["note"][:2]
        
        for p_raw in (t1 + t2):
            p_name = base_name(p_raw)
            if p_name not in stats:
                # 📌 남복, 여복, 혼복, 잡복 카운트용 변수 추가
                stats[p_name] = {
                    "League": league, 
                    "1R": "-", "2R": "-", "3R": "-", "4R": "-", 
                    "남복": 0, "여복": 0, "혼복": 0, "잡복": 0,
                    "Total": 0
                }
            
            if "1R" in r_num: stats[p_name]["1R"] = match_type
            elif "2R" in r_num: stats[p_name]["2R"] = match_type
            elif "3R" in r_num: stats[p_name]["3R"] = match_type
            elif "4R" in r_num: stats[p_name]["4R"] = match_type
            
            # 📌 횟수 증가 로직
            if match_type in ["남복", "여복", "혼복", "잡복"]:
                stats[p_name][match_type] += 1
                
            stats[p_name]["Total"] += 1
    
    data = []
    for name, info in stats.items():
        data.append({
            "리그": info["League"], "이름": name, 
            "1R": info["1R"], "2R": info["2R"], "3R": info["3R"], "4R": info["4R"], 
            # 📌 새로 생성한 4개의 열을 '총합' 앞에 배치
            "남복": info["남복"], "여복": info["여복"], "혼복": info["혼복"], "잡복": info["잡복"], 
            "총합": info["Total"]
        })
    return pd.DataFrame(data).sort_values(by=["리그", "이름"])

# ==========================================
# 2. 스트림릿(Streamlit) 화면 구성
# ==========================================
st.set_page_config(page_title="TELA Tennis Match", page_icon="🎾", layout="wide")

st.title("🎾 TELA CLUB Random Match_WEB(v1.06)")
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
    
    display_data = []
    for d in data:
        display_data.append({
            "라운드": d["round"],
            "리그": d["league"],
            "팀1(1)": d['team1'][0],
            "팀1(2)": d['team1'][1],
            "팀2(1)": d['team2'][0],
            "팀2(2)": d['team2'][1],
            "비고": d["note"]
        })
    df_matches = pd.DataFrame(display_data)

    tab1, tab2 = st.tabs(["📋 대진표", "📊 출전 현황"])
    
    with tab1:
        st.subheader("경기 매치업")
        
        def highlight_rows(row):
            if "A리그" in row["리그"]:
                return ['background-color: #C8E6C9; color: black'] * len(row)
            else:
                return ['background-color: #BBDEFB; color: black'] * len(row)

        st.dataframe(df_matches.style.apply(highlight_rows, axis=1), use_container_width=True, height=600)

    with tab2:
        st.subheader("선수별 출전 기록")
        df_stats = calculate_stats(data)
        
        def highlight_stats(val):
            if isinstance(val, int):
                if val < 3: return 'background-color: #FFCDD2; color: black'
                if val == 3: return 'background-color: #E8F5E9; color: black'
                if val >= 4: return 'background-color: #FFF9C4; color: black'
            return ''

        st.dataframe(df_stats.style.applymap(highlight_stats, subset=["총합"]), use_container_width=True)

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
