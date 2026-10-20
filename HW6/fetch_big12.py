"""Download Big 12 conference games for 2025 and 2026 from ESPN's public
scoreboard and write big12.csv. Run from hw6/; rerun after each weekend.
Unplayed games are kept (scores NaN) so they can be predicted.
Standard library only; same output as fetch_big12.m."""
import csv, json, urllib.request

URL = ('https://site.api.espn.com/apis/site/v2/sports/football/college-football/'
       'scoreboard?seasontype=2&groups=4&limit=200')
rows = []
for season in (2025, 2026):
    for w in range(1, 16):
        with urllib.request.urlopen(f'{URL}&dates={season}&week={w}', timeout=30) as f:
            d = json.load(f)
        for ev in d.get('events', []):
            c = ev['competitions'][0]
            cp = c['competitors']
            if not all(str(x['team'].get('conferenceId', '')) == '4' for x in cp):
                continue                                    # Big 12 vs Big 12 only
            H = next(x for x in cp if x['homeAway'] == 'home')
            A = next(x for x in cp if x['homeAway'] == 'away')
            done = int(bool(c['status']['type']['completed']))
            sc = lambda x: int(float(x['score'])) if done else 'NaN'
            row = (season, w, ev['date'][:10], H['team']['location'], sc(H),
                   A['team']['location'], sc(A), int(bool(c['neutralSite'])), done)
            if row not in rows:
                rows.append(row)

with open('big12.csv', 'w', newline='') as f:
    out = csv.writer(f)
    out.writerow(['season', 'week', 'date', 'home', 'home_pts', 'away', 'away_pts',
                  'neutral', 'completed'])
    out.writerows(rows)
n25 = sum(r[0] == 2025 for r in rows)
n26 = sum(r[0] == 2026 and r[8] == 1 for r in rows)
print(f'{len(rows)} games written to big12.csv (2025: {n25}, 2026 played: {n26}, '
      f'2026 to play: {len(rows) - n25 - n26})')
