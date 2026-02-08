#!/usr/bin/env python3
"""
Generate plots and EDA analysis from saved results.json files.

Usage:
    # Quick plots (aggregated only)
    python scripts/plot_results.py results.json
    
    # FULL EDA with individual plots for each test case + statistics
    python scripts/plot_results.py results.json --eda
    
    # Specify output directory
    python scripts/plot_results.py results.json --eda --output ./my_analysis
    
    # Limit individual problem plots
    python scripts/plot_results.py results.json --eda --max-individual 50
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots and EDA analysis from saved results JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick aggregate plots
  python scripts/plot_results.py path/to/results.json
  
  # Full EDA with individual plots + statistics  
  python scripts/plot_results.py path/to/results.json --eda
  
  # EDA with custom output and limited individual plots
  python scripts/plot_results.py path/to/results.json --eda -o ./analysis --max-individual 50
        """,
    )
    
    parser.add_argument(
        "results_path",
        type=str,
        help="Path to results.json or combined_results.json"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: plots/ or eda/ in same dir)"
    )
    
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Name for plot titles (default: inferred from path)"
    )
    
    parser.add_argument(
        "--combined", "-c",
        action="store_true",
        help="Treat input as combined_results.json (multiple variations)"
    )
    
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run FULL EDA: individual plots for each test case + aggregate statistics"
    )
    
    parser.add_argument(
        "--max-individual",
        type=int,
        default=100,
        help="Maximum number of individual problem plots to generate (default: 100)"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results_path)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    
    output_dir = Path(args.output) if args.output else None
    
    if args.eda:
        # Full EDA mode
        from p_sparsity.case_studies.eda import run_full_eda, EDAConfig
        
        config = EDAConfig(max_individual_plots=args.max_individual)
        run_full_eda(results_path, output_dir, config)
    
    elif args.combined:
        # Combined results with multiple variations
        from p_sparsity.case_studies.visualization import generate_all_plots
        generate_all_plots(results_path, output_dir)
    
    else:
        # Single variation - quick plots
        from p_sparsity.case_studies.visualization import plot_from_results_json
        plot_from_results_json(
            results_path,
            output_dir=output_dir,
            variation_name=args.name,
        )


if __name__ == "__main__":
    main()
