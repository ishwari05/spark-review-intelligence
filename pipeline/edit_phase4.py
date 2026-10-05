import os

filepath = 'run_phase4_experiments.py'
with open(filepath, 'r') as f:
    content = f.read()

# Replace Phase 3 references
content = content.replace('Phase 3', 'Phase 4')
content = content.replace('phase3', 'phase4')
content = content.replace('PHASE3', 'PHASE4')

# Edit prepare_data
old_prepare = """def prepare_data(raw_df):
    df = raw_df.dropna(subset=["review"])
    df = df.dropDuplicates(["review"])
    df = df.filter(F.length(F.trim(F.col("review"))) > 0)
    return df"""

new_prepare = """def prepare_data(raw_df):
    # Phase 4 modification: Combine title and review for richer context
    df = raw_df.dropna(subset=["title", "review"])
    df = df.withColumn("review", F.concat_ws(" ", F.col("title"), F.col("review")))
    df = df.dropDuplicates(["review"])
    df = df.filter(F.length(F.trim(F.col("review"))) > 0)
    return df"""

content = content.replace(old_prepare, new_prepare)

# We want only one experiment (the best Phase 3 LR model)
# Let's replace the whole main logic, or just let it run all models again to see how they do with combined text.
# Actually, the user asked to combine title+content. Running all Phase 3 models on this new text might be fine, but we can just let it run the experiments. It's safe.

with open(filepath, 'w') as f:
    f.write(content)

print("Edits complete.")
