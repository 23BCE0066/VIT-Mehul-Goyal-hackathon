"""
Module B: Synthetic Wholesale Banking Book Loader
Defines and loads multi-asset synthetic wholesale portfolio:
- Commercial Term Loans & Credit Facilities
- Corporate Senior Unsecured Notes & Bonds
- Rates, FX, Credit & Commodity Derivatives
Computes baseline mark-to-market and Basel regulatory risk provisions.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class WholesalePortfolioBook:
    def __init__(self, csv_path: Optional[str] = None):
        if csv_path is None:
            self.csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "wholesale_portfolio.csv"
        else:
            self.csv_path = Path(csv_path)

    def load_portfolio(self) -> pd.DataFrame:
        """Loads and calculates initial baseline credit metrics."""
        df = pd.read_csv(self.csv_path)

        # Baseline Expected Loss (EL = EAD * PD * LGD)
        df["baseline_expected_loss_usd"] = df["notional_usd"] * df["baseline_pd"] * df["baseline_lgd"]
        
        # Approximate baseline valuation
        df["baseline_market_value_usd"] = df["notional_usd"] - df["baseline_expected_loss_usd"]
        return df

    def get_summary_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """Aggregates portfolio composition by asset class and rating."""
        if df is None:
            df = self.load_portfolio()

        total_notional = float(df["notional_usd"].sum())
        total_baseline_el = float(df["baseline_expected_loss_usd"].sum())
        el_rate_bps = (total_baseline_el / total_notional) * 10000

        by_asset = df.groupby("asset_type")["notional_usd"].sum().to_dict()
        by_sector = df.groupby("borrower_sector")["notional_usd"].sum().to_dict()

        return {
            "total_notional_usd": total_notional,
            "total_baseline_expected_loss": total_baseline_el,
            "expected_loss_bps": round(el_rate_bps, 1),
            "asset_breakdown": by_asset,
            "sector_breakdown": by_sector,
            "total_positions": len(df)
        }


if __name__ == "__main__":
    book = WholesalePortfolioBook()
    df = book.load_portfolio()
    stats = book.get_summary_statistics(df)
    print("--- SYNTHETIC WHOLESALE PORTFOLIO SUMMARY ---")
    print(f"Total Notional: ${stats['total_notional_usd']:,.0f}")
    print(f"Total Baseline Expected Loss: ${stats['total_baseline_expected_loss']:,.0f} ({stats['expected_loss_bps']} bps)")
    print(f"Total Positions: {stats['total_positions']}")
