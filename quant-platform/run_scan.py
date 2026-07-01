"""
Local scan runner — shows top setups even below the 85 hard threshold,
so you can see what the market is offering today.

Usage:
    python run_scan.py            # US market
    python run_scan.py india      # India market
"""
import sys
from scanner.engine import Scanner

market = "INDIA" if (len(sys.argv) > 1 and sys.argv[1].lower() == "india") else "US"

scanner = Scanner(market=market)
# Lower the gate so we can see the ranked board even on quiet days
scanner._min_score = 0.0
opps = scanner.run(save_to_db=False)

if not opps:
    print(f"\nNo signals generated at all for {market} today.")
    print("Every strategy returned None — market conditions don't match any setup.")
    sys.exit(0)

print(f"\n{'='*100}")
print(f"{market} SCAN — {len(opps)} signals (ranked by score)")
print(f"{'='*100}")
print(f"{'TICKER':<10}{'STRATEGY':<18}{'SCORE':>6}{'ENTRY':>10}{'STOP':>10}"
      f"{'T1':>10}{'T2':>10}{'RR_T2':>7}{'RS':>6}")
print("-"*100)

for o in opps[:25]:
    print(f"{o.ticker:<10}{o.strategy:<18}{o.score:>6.0f}"
          f"{o.ideal_entry:>10.2f}{o.stop_loss:>10.2f}"
          f"{o.target_1:>10.2f}{o.target_2:>10.2f}"
          f"{o.rr_ratio_t2:>7.1f}{o.rs_rank:>6.0f}")

# Highlight the ones that clear the real 85 quality bar
tradeable = [o for o in opps if o.score >= 85]
print(f"\n{'='*100}")
if tradeable:
    print(f"TRADEABLE (score >= 85): {len(tradeable)} setups")
    for o in tradeable:
        print(f"  {o.ticker} ({o.strategy}) score={o.score:.0f} "
              f"entry={o.ideal_entry:.2f} stop={o.stop_loss:.2f} "
              f"RR_T2={o.rr_ratio_t2:.1f}")
else:
    print("TRADEABLE (score >= 85): NONE today — stay in cash / wait for better setups.")
print(f"{'='*100}")
