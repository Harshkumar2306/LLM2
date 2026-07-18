import sys
import os
from collections import Counter
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trainer.data_manager import WeightedDistributedSampler

def test_sampler():
    print("Testing WeightedDistributedSampler Distribution...\n")
    
    # Mock datasets matching our 15GB targets
    dataset_lengths = [
        9000000,  # fineweb (60%)
        3000000,  # stack (20%)
        1200000,  # wikipedia (8%)
        1200000,  # api_docs (8%)
        600000,   # books (4%)
    ]
    weights = [0.60, 0.20, 0.08, 0.08, 0.04]
    
    sampler = WeightedDistributedSampler(dataset_lengths, weights, num_replicas=1, rank=0)
    
    # We will draw 100,000 samples and count where they land
    samples_to_draw = 100000
    counts = [0, 0, 0, 0, 0]
    
    # Helper to figure out which dataset an absolute index belongs to
    def get_dataset_idx(absolute_idx):
        for i in range(len(sampler.offsets)-1):
            if sampler.offsets[i] <= absolute_idx < sampler.offsets[i+1]:
                return i
        return len(dataset_lengths) - 1

    print(f"Drawing {samples_to_draw:,} random batches...")
    
    iterator = iter(sampler)
    for _ in range(samples_to_draw):
        idx = next(iterator)
        ds_idx = get_dataset_idx(idx)
        counts[ds_idx] += 1
        
    print("\nResults:")
    print("-" * 50)
    labels = ["FineWeb", "Stack", "Wikipedia", "API Docs", "Books"]
    
    for label, count, target_weight in zip(labels, counts, weights):
        observed_weight = count / samples_to_draw
        error = abs(observed_weight - target_weight)
        print(f"{label:<12}: Target: {target_weight*100:5.2f}% | Observed: {observed_weight*100:5.2f}% | Error: {error*100:.2f}%")
        
    # Assert errors are < 1% (probabilistic, but 100k samples should be very tight)
    for count, target_weight in zip(counts, weights):
        observed_weight = count / samples_to_draw
        assert abs(observed_weight - target_weight) < 0.01, f"Sampling error too high: {observed_weight} vs {target_weight}"
        
    print("\n[SUCCESS] Sampler distribution matches configured weights!")

if __name__ == "__main__":
    test_sampler()
