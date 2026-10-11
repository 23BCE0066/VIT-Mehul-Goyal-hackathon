"""
Module B: Strategic Portfolio Stress Testing Engine
Subscribes to Event Classification and Impact Score from the AI/NLP Risk Engine.
When high-impact events occur (Impact Score >= 7):
- Translates unstructured event classifications into macroeconomic and credit shock vectors
- Simulates multi-asset valuation haircuts:
  * Commercial Loans: Credit migration, PD multiplier spike, collateral haircut (LGD expansion)
  * Corporate Bonds: Yield curve shift, credit spread widening (Modified Duration price drop)
  * Derivatives: Counterparty risk spike, mark-to-market basis volatility
- Quantifies Portfolio Pre- vs. Post-Stress values, Capital Degradation, and Stressed Expected Loss.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from src.module_b.synthetic_book import WholesalePortfolioBook


class StrategicStressTester:
    def __init__(self, portfolio_book: Optional[WholesalePortfolioBook] = None):
        self.book = portfolio_book or WholesalePortfolioBook()

        # Institutional shock parameter matrix mapped to event categories
        self.shock_transmission_models = {
            "Geopolitical": {
                "name": "Severe Geopolitical & Energy Shock",
                "spread_widening_bps": 160.0,
                "rate_shock_pct": 0.0075,      # +75 bps flight to safety/inflation
                "pd_multiplier": 1.75,          # 75% increase in baseline default rate
                "lgd_expansion_pct": 0.15,      # 15% reduction in recovery values
                "sector_sensitivities": {
                    "Transportation": 2.2,
                    "Energy & Utilities": 0.4,   # Energy hedge benefits
                    "Consumer Discretionary": 1.8,
                    "Commercial Real Estate": 1.5
                }
            },
            "Macroeconomic": {
                "name": "Aggressive Rate Tightening & Liquidity Contraction",
                "spread_widening_bps": 120.0,
                "rate_shock_pct": 0.0200,      # +200 bps policy rate hike
                "pd_multiplier": 1.90,          # Borrowers struggle with debt service
                "lgd_expansion_pct": 0.10,
                "sector_sensitivities": {
                    "Commercial Real Estate": 2.5,
                    "Financial Services": 1.3,
                    "Technology": 1.4,
                    "Healthcare": 0.8
                }
            },
            "Credit Event": {
                "name": "Contagion Default & Syndicate Downgrade",
                "spread_widening_bps": 280.0,
                "rate_shock_pct": 0.0025,
                "pd_multiplier": 3.10,          # 310% surge in default probabilities
                "lgd_expansion_pct": 0.35,      # Collateral fire-sale depression
                "sector_sensitivities": {
                    "Commercial Real Estate": 3.2,
                    "High Yield Corporates": 2.8,
                    "Banking & Sovereigns": 1.6,
                    "Consumer Discretionary": 1.9
                }
            },
            "Merger/Acquisition": {
                "name": "Leverage Re-gearing & Integration Friction",
                "spread_widening_bps": 40.0,
                "rate_shock_pct": 0.0010,
                "pd_multiplier": 1.15,
                "lgd_expansion_pct": 0.05,
                "sector_sensitivities": {
                    "Technology": 1.2,
                    "Financial Services": 1.1
                }
            },
            "Product Launch": {
                "name": "Competitive Disruption Displacement",
                "spread_widening_bps": 20.0,
                "rate_shock_pct": 0.0000,
                "pd_multiplier": 1.05,
                "lgd_expansion_pct": 0.02,
                "sector_sensitivities": {
                    "Technology": 1.2,
                    "Healthcare": 1.1
                }
            }
        }

    def evaluate_event_trigger(self, signal: Dict) -> bool:
        """Determines if incoming signal qualifies as a stress test trigger (Impact >= 7)."""
        impact = signal.get("impact_score", 0)
        return impact >= 7

    def run_stress_test(self, event_category: str, impact_score: int, custom_portfolio_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Executes granular stress simulation across portfolio positions.
        Scales shock intensity relative to impact score (1 to 10).
        """
        df = custom_portfolio_df.copy() if custom_portfolio_df is not None else self.book.load_portfolio()
        shock_params = self.shock_transmission_models.get(
            event_category, self.shock_transmission_models["Macroeconomic"]
        )

        # Scale factor based on impact score: 7 -> 1.0x, 10 -> 1.6x
        severity_scaling = max(0.5, (impact_score - 5.0) / 3.0)

        rate_shock = shock_params["rate_shock_pct"] * severity_scaling
        spread_shock = (shock_params["spread_widening_bps"] / 10000.0) * severity_scaling
        base_pd_mult = 1.0 + ((shock_params["pd_multiplier"] - 1.0) * severity_scaling)
        base_lgd_exp = shock_params["lgd_expansion_pct"] * severity_scaling

        stressed_rows = []
        for _, row in df.iterrows():
            sec = row["borrower_sector"]
            sec_beta = shock_params["sector_sensitivities"].get(sec, 1.0)

            # Stressed PD & LGD
            st_pd = min(1.0, row["baseline_pd"] * (base_pd_mult * sec_beta))
            st_lgd = min(1.0, row["baseline_lgd"] * (1.0 + base_lgd_exp * sec_beta))
            st_expected_loss = row["notional_usd"] * st_pd * st_lgd

            # Valuation impact based on asset class mechanics:
            asset_type = row["asset_type"]
            dur = row["duration_years"]
            notional = row["notional_usd"]

            if asset_type == "Commercial Loan":
                # Loans revalued via increase in credit risk provisions + spread impact
                haircut_pct = (st_pd - row["baseline_pd"]) * st_lgd + (spread_shock * sec_beta * (dur * 0.5))
            elif asset_type == "Corporate Bond":
                # Bonds priced via Modified Duration: delta_P / P ~ -Duration * (delta_y + delta_spread)
                haircut_pct = (dur * (rate_shock + spread_shock * sec_beta))
            else:
                # Derivatives stressed via volatility and counterparty probability of default
                haircut_pct = (rate_shock * dur * 1.5) + (st_pd * 0.5)

            haircut_pct = min(max(haircut_pct, 0.0), 0.70)  # Capped realistic loss
            st_market_val = notional * (1.0 - haircut_pct)
            loss_amount = row["baseline_market_value_usd"] - st_market_val

            stressed_rows.append({
                "asset_id": row["asset_id"],
                "facility_name": row["facility_name"],
                "asset_type": asset_type,
                "borrower_sector": sec,
                "notional_usd": notional,
                "duration_years": dur,
                "baseline_mv_usd": row["baseline_market_value_usd"],
                "stressed_mv_usd": round(st_market_val, 2),
                "valuation_loss_usd": round(loss_amount, 2),
                "loss_pct": round(haircut_pct * 100, 2),
                "baseline_pd_pct": round(row["baseline_pd"] * 100, 2),
                "stressed_pd_pct": round(st_pd * 100, 2),
                "baseline_el_usd": row["baseline_expected_loss_usd"],
                "stressed_el_usd": round(st_expected_loss, 2)
            })

        st_df = pd.DataFrame(stressed_rows)
        total_base_mv = float(st_df["baseline_mv_usd"].sum())
        total_st_mv = float(st_df["stressed_mv_usd"].sum())
        total_loss = float(st_df["valuation_loss_usd"].sum())
        total_base_el = float(st_df["baseline_el_usd"].sum())
        total_st_el = float(st_df["stressed_el_usd"].sum())

        return {
            "scenario_name": shock_params["name"],
            "event_category": event_category,
            "impact_score": impact_score,
            "scaling_factor": round(severity_scaling, 2),
            "total_pre_stress_mv": total_base_mv,
            "total_post_stress_mv": total_st_mv,
            "total_portfolio_loss_usd": total_loss,
            "portfolio_drawdown_pct": round((total_loss / total_base_mv) * 100, 2),
            "pre_stress_expected_loss": total_base_el,
            "post_stress_expected_loss": total_st_el,
            "expected_loss_surge_pct": round(((total_st_el - total_base_el) / total_base_el) * 100, 2),
            "detailed_asset_impact": st_df
        }


if __name__ == "__main__":
    stress_tester = StrategicStressTester()
    res = stress_tester.run_stress_test("Credit Event", impact_score=9)
    print("\n--- STRESS TEST SIMULATION RESULTS ---")
    print(f"Scenario: {res['scenario_name']}")
    print(f"Pre-Stress Value:  ${res['total_pre_stress_mv']:,.0f}")
    print(f"Post-Stress Value: ${res['total_post_stress_mv']:,.0f}")
    print(f"Total Loss:        ${res['total_portfolio_loss_usd']:,.0f} (-{res['portfolio_drawdown_pct']}%)")
    print(f"Expected Loss Jump: ${res['pre_stress_expected_loss']:,.0f} -> ${res['post_stress_expected_loss']:,.0f} (+{res['expected_loss_surge_pct']}%)")
