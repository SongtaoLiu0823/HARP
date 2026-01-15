import json
import argparse

def calculate_weighted_average(result_dict):
    # Extract exact match scores
    exact_matches = {
        "leaderboard_math_algebra_hard": result_dict["results"]["leaderboard_math_algebra_hard"]["exact_match,none"],
        "leaderboard_math_counting_and_prob_hard": result_dict["results"]["leaderboard_math_counting_and_prob_hard"]["exact_match,none"],
        "leaderboard_math_geometry_hard": result_dict["results"]["leaderboard_math_geometry_hard"]["exact_match,none"],
        "leaderboard_math_intermediate_algebra_hard": result_dict["results"]["leaderboard_math_intermediate_algebra_hard"]["exact_match,none"],
        "leaderboard_math_num_theory_hard": result_dict["results"]["leaderboard_math_num_theory_hard"]["exact_match,none"],
        "leaderboard_math_prealgebra_hard": result_dict["results"]["leaderboard_math_prealgebra_hard"]["exact_match,none"],
        "leaderboard_math_precalculus_hard": result_dict["results"]["leaderboard_math_precalculus_hard"]["exact_match,none"],
    }

    # Extract sample counts
    sample_counts = {
        "leaderboard_math_algebra_hard": result_dict["n-samples"]["leaderboard_math_algebra_hard"]["original"],
        "leaderboard_math_counting_and_prob_hard": result_dict["n-samples"]["leaderboard_math_counting_and_prob_hard"]["original"],
        "leaderboard_math_geometry_hard": result_dict["n-samples"]["leaderboard_math_geometry_hard"]["original"],
        "leaderboard_math_intermediate_algebra_hard": result_dict["n-samples"]["leaderboard_math_intermediate_algebra_hard"]["original"],
        "leaderboard_math_num_theory_hard": result_dict["n-samples"]["leaderboard_math_num_theory_hard"]["original"],
        "leaderboard_math_prealgebra_hard": result_dict["n-samples"]["leaderboard_math_prealgebra_hard"]["original"],
        "leaderboard_math_precalculus_hard": result_dict["n-samples"]["leaderboard_math_precalculus_hard"]["original"],
    }

    # Calculate total samples
    total_samples = sum(sample_counts.values())

    # Calculate weighted average
    weighted_average = sum(
        exact_matches[task] * sample_counts[task] for task in exact_matches
    ) / total_samples

    return weighted_average

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True, help="Path to the results JSON file")
    args = parser.parse_args()

    # Load the results JSON file
    with open(args.result_path, 'r', encoding='utf-8') as f:
        result_dict = json.load(f)

    # Calculate weighted average
    weighted_average = calculate_weighted_average(result_dict)
    print(f"Weighted Average Exact Match: {weighted_average:.6f}")

if __name__ == "__main__":
    main()

