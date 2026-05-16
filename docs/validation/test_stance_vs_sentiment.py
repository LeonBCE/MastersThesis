"""
Test: Stance (NLI) vs Sentiment on 85 diagnostic political tweets.
Run in Colab or locally with: pip install transformers torch pandas
"""
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Diagnostic AI-assisted tweets covering the main macro-topic areas.
# Format: (tweet, topic, expected_political_lean, expected_emotion)
# expected_political_lean: "conservative", "progressive", "neutral/ambiguous"
# expected_emotion: "angry", "positive", "neutral", "sarcastic"

tweets_data = [
    # ============ DEMOCRACY_LAW — Courts (Topic 9) ============
    ("The Supreme Court is protecting our constitutional rights. Thank God for conservative justices.", "DemLaw_Courts", "conservative", "positive"),
    ("Clarence Thomas needs to be impeached. His corruption is destroying the court.", "DemLaw_Courts", "progressive", "angry"),
    ("The Supreme Court ruled 6-3 on the immunity case today.", "DemLaw_Courts", "neutral", "neutral"),
    ("Conservative judges are dismantling democracy one ruling at a time.", "DemLaw_Courts", "progressive", "angry"),
    ("Finally a Supreme Court that respects the constitution as written!", "DemLaw_Courts", "conservative", "positive"),
    ("Merrick Garland is the worst AG in American history. Completely weaponized the DOJ.", "DemLaw_Courts", "conservative", "angry"),
    ("Merrick Garland needs to do his job and prosecute these criminals.", "DemLaw_Courts", "progressive", "angry"),
    ("The DOJ investigation has been thorough and fair regardless of what anyone says.", "DemLaw_Courts", "neutral", "neutral"),
    ("SCOTUS just handed corporations another win over regular people.", "DemLaw_Courts", "progressive", "angry"),
    ("Our judges are upholding law and order. This is what America needs.", "DemLaw_Courts", "conservative", "positive"),

    # ============ DEMOCRACY_LAW — Felon (Topic 6) ============
    ("I will proudly vote for a convicted felon over a corrupt Democrat any day.", "DemLaw_Felon", "conservative", "angry"),
    ("We're really about to elect a convicted felon as president. What a country.", "DemLaw_Felon", "progressive", "sarcastic"),
    ("Trump was convicted on 34 felony counts. That's just a fact.", "DemLaw_Felon", "neutral", "neutral"),
    ("The conviction was a political witch hunt. Everyone knows it.", "DemLaw_Felon", "conservative", "angry"),
    ("A convicted felon should not be allowed to run for president. Period.", "DemLaw_Felon", "progressive", "angry"),
    ("Trump's conviction proves no one is above the law in America.", "DemLaw_Felon", "progressive", "positive"),
    ("The jury was rigged in a blue city. This conviction means nothing.", "DemLaw_Felon", "conservative", "angry"),
    ("34 felony counts and people still support him. I can't understand it.", "DemLaw_Felon", "progressive", "angry"),
    ("Trump was found guilty by a jury of his peers. That's how justice works.", "DemLaw_Felon", "progressive", "neutral"),
    ("This trial was election interference pure and simple.", "DemLaw_Felon", "conservative", "angry"),

    # ============ ECONOMY (Topic 3) ============
    ("Grocery prices are insane. I can barely afford to feed my family anymore.", "Economy", "neutral", "angry"),
    ("Bidenomics is working. Unemployment is at record lows.", "Economy", "progressive", "positive"),
    ("Inflation is destroying the middle class and Biden doesn't care.", "Economy", "conservative", "angry"),
    ("Gas prices are finally coming down. Thank you Mr. President.", "Economy", "progressive", "positive"),
    ("The economy is rigged for the billionaires. Both parties are guilty.", "Economy", "neutral", "angry"),
    ("Trump's tax cuts saved my small business. We need him back.", "Economy", "conservative", "positive"),
    ("The national debt is out of control under both parties.", "Economy", "neutral", "angry"),
    ("Corporate greed is the real cause of inflation, not government spending.", "Economy", "progressive", "angry"),
    ("My 401k is doing great under Biden. Facts don't care about feelings.", "Economy", "progressive", "sarcastic"),
    ("We need tariffs to protect American workers from cheap Chinese goods.", "Economy", "conservative", "neutral"),

    # ============ IMMIGRATION (Topic 1) ============
    ("The border is wide open and Biden refuses to do anything about it.", "Immigration", "conservative", "angry"),
    ("Immigrants built this country. We should welcome them, not cage them.", "Immigration", "progressive", "positive"),
    ("Illegal immigration is destroying our communities and driving down wages.", "Immigration", "conservative", "angry"),
    ("The border crisis is manufactured fear. Net migration hasn't changed that much.", "Immigration", "progressive", "neutral"),
    ("We need to secure the border AND provide a path to citizenship.", "Immigration", "neutral", "neutral"),
    ("Deport them all. We don't have room or resources for millions of illegals.", "Immigration", "conservative", "angry"),
    ("These are families seeking asylum. That's legal under international law.", "Immigration", "progressive", "neutral"),
    ("Fentanyl is pouring across the border and killing our kids.", "Immigration", "conservative", "angry"),
    ("Both parties have failed on immigration for decades. Nothing new.", "Immigration", "neutral", "angry"),
    ("Build the wall. Finish it. Now.", "Immigration", "conservative", "angry"),

    # ============ FOREIGN POLICY — Israel (Topic 2) ============
    ("Israel has the right to defend itself against Hamas terrorism.", "ForeignPolicy_Israel", "conservative", "neutral"),
    ("What's happening in Gaza is genocide. Stop funding Israel.", "ForeignPolicy_Israel", "progressive", "angry"),
    ("The civilian death toll in Gaza is heartbreaking. We need a ceasefire now.", "ForeignPolicy_Israel", "progressive", "angry"),
    ("Hamas started this war. Israel is finishing it. That's how war works.", "ForeignPolicy_Israel", "conservative", "neutral"),
    ("I stand with Israel. Always have, always will.", "ForeignPolicy_Israel", "conservative", "positive"),
    ("Cutting off aid to Israel would be a catastrophic foreign policy mistake.", "ForeignPolicy_Israel", "conservative", "neutral"),
    ("Free Palestine. End the occupation.", "ForeignPolicy_Israel", "progressive", "angry"),
    ("Both sides are suffering. We need peace not more weapons.", "ForeignPolicy_Israel", "neutral", "neutral"),
    ("Netanyahu is a war criminal and Biden is complicit.", "ForeignPolicy_Israel", "progressive", "angry"),
    ("Iran is behind Hamas and Hezbollah. We need to support our ally Israel.", "ForeignPolicy_Israel", "conservative", "neutral"),

    # ============ FOREIGN POLICY — China (Topic 40) ============
    ("China is our biggest threat. We need to decouple our economies now.", "ForeignPolicy_China", "conservative", "angry"),
    ("The CCP is a communist regime that threatens global freedom.", "ForeignPolicy_China", "conservative", "angry"),
    ("We need diplomacy with China, not another cold war.", "ForeignPolicy_China", "progressive", "neutral"),
    ("Chinese spy balloons flying over our country and Biden did nothing.", "ForeignPolicy_China", "conservative", "angry"),
    ("The real threat isn't China, it's our own oligarchs.", "ForeignPolicy_China", "progressive", "angry"),

    # ============ SOCIAL ISSUES — Abortion (Topic 18) ============
    ("My body, my choice. Roe v Wade should never have been overturned.", "SocialIssues_Abortion", "progressive", "angry"),
    ("Every life is sacred from conception. Proud to be pro-life.", "SocialIssues_Abortion", "conservative", "positive"),
    ("Women are dying because they can't access basic reproductive healthcare.", "SocialIssues_Abortion", "progressive", "angry"),
    ("Abortion is murder. There's no other way to say it.", "SocialIssues_Abortion", "conservative", "angry"),
    ("The abortion bans are wildly unpopular. Even red states are voting against them.", "SocialIssues_Abortion", "progressive", "neutral"),

    # ============ SOCIAL ISSUES — Race (Topic 7) ============
    ("Systemic racism is real and if you deny it you're part of the problem.", "SocialIssues_Race", "progressive", "angry"),
    ("DEI is just racism with extra steps. Hire based on merit.", "SocialIssues_Race", "conservative", "angry"),
    ("The black community deserves better than empty promises from Democrats.", "SocialIssues_Race", "neutral", "angry"),
    ("Woke culture is destroying America. People are tired of it.", "SocialIssues_Race", "conservative", "angry"),
    ("Black voters aren't a monolith. Stop treating us like one.", "SocialIssues_Race", "neutral", "angry"),

    # ============ SOCIAL ISSUES — Guns (Topic 23) ============
    ("The second amendment is non-negotiable. Shall not be infringed.", "SocialIssues_Guns", "conservative", "angry"),
    ("How many kids have to die before we pass common sense gun laws?", "SocialIssues_Guns", "progressive", "angry"),
    ("More guns means more gun deaths. The data is clear.", "SocialIssues_Guns", "progressive", "neutral"),
    ("Gun control only disarms law-abiding citizens. Criminals don't follow laws.", "SocialIssues_Guns", "conservative", "neutral"),
    ("Another mass shooting and Congress does nothing. Again.", "SocialIssues_Guns", "progressive", "angry"),

    # ============ SOCIAL ISSUES — Climate (Topic 30) ============
    ("Climate change is the existential threat of our time. Act now.", "SocialIssues_Climate", "progressive", "angry"),
    ("Climate change is a hoax to control the economy. Follow the money.", "SocialIssues_Climate", "conservative", "angry"),
    ("We need to invest in renewable energy for our children's future.", "SocialIssues_Climate", "progressive", "positive"),
    ("The climate has always changed. Humans aren't causing it.", "SocialIssues_Climate", "conservative", "neutral"),
    ("Electric vehicles are the future whether you like it or not.", "SocialIssues_Climate", "progressive", "neutral"),

    # ============ DOMESTIC POLICY — Healthcare (Topic 19) ============
    ("Medicare for all is the only moral healthcare system.", "DomesticPolicy_Healthcare", "progressive", "angry"),
    ("Government healthcare means worse care and longer wait times.", "DomesticPolicy_Healthcare", "conservative", "angry"),
    ("I went bankrupt from medical bills in the richest country on earth.", "DomesticPolicy_Healthcare", "neutral", "angry"),
    ("The ACA saved my life. Pre-existing conditions would have killed me.", "DomesticPolicy_Healthcare", "progressive", "positive"),
    ("Healthcare costs are out of control. Something has to change.", "DomesticPolicy_Healthcare", "neutral", "angry"),

    # ============ DOMESTIC POLICY — Covid (Topic 12) ============
    ("The covid vaccine saved millions of lives. Trust the science.", "DomesticPolicy_Covid", "progressive", "positive"),
    ("Vaccine mandates were government overreach. My body my choice.", "DomesticPolicy_Covid", "conservative", "angry"),
    ("Fauci lied and people died. We need accountability.", "DomesticPolicy_Covid", "conservative", "angry"),
    ("Covid is still killing people and nobody cares anymore.", "DomesticPolicy_Covid", "progressive", "angry"),
    ("The lockdowns destroyed more lives than they saved.", "DomesticPolicy_Covid", "conservative", "angry"),
]

tweets = [t[0] for t in tweets_data]
topics = [t[1] for t in tweets_data]
expected_lean = [t[2] for t in tweets_data]
expected_emotion = [t[3] for t in tweets_data]

print(f"Total tweets: {len(tweets)}")
print(f"Topics: {pd.Series(topics).value_counts().to_dict()}")

# ── Load the EXACT stance model from the pipeline ──
print("\nLoading stance model (DeBERTa NLI)...")
STANCE_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
stance_tokenizer = AutoTokenizer.from_pretrained(STANCE_MODEL)
stance_model = AutoModelForSequenceClassification.from_pretrained(STANCE_MODEL)
stance_model.eval()

# ── Load a sentiment model ──
print("Loading sentiment model (twitter-roberta)...")
SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sent_tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL)
sent_model.eval()
# Labels: 0=negative, 1=neutral, 2=positive

# ── Run stance detection (exact same method as pipeline) ──
print("\nRunning stance detection...")
cons_hyp = "This text expresses conservative right-wing political views."
prog_hyp = "This text expresses progressive left-wing political views."

stance_scores = []
cons_raw = []
prog_raw = []

with torch.no_grad():
    for tweet in tweets:
        # Conservative hypothesis
        enc_c = stance_tokenizer(tweet, cons_hyp, return_tensors="pt",
                                  truncation=True, max_length=512)
        out_c = stance_model(**enc_c)
        p_cons = F.softmax(out_c.logits.float(), dim=1)[0, 2].item()

        # Progressive hypothesis
        enc_p = stance_tokenizer(tweet, prog_hyp, return_tensors="pt",
                                  truncation=True, max_length=512)
        out_p = stance_model(**enc_p)
        p_prog = F.softmax(out_p.logits.float(), dim=1)[0, 2].item()

        stance_scores.append(p_cons - p_prog)
        cons_raw.append(p_cons)
        prog_raw.append(p_prog)

# ── Run sentiment analysis ──
print("Running sentiment analysis...")
sent_scores = []
sent_labels = []
sent_neg = []
sent_neu = []
sent_pos = []

with torch.no_grad():
    for tweet in tweets:
        enc = sent_tokenizer(tweet, return_tensors="pt",
                              truncation=True, max_length=512)
        out = sent_model(**enc)
        probs = F.softmax(out.logits.float(), dim=1)[0]
        # Score from -1 (negative) to +1 (positive): weighted sum
        score = -1 * probs[0].item() + 0 * probs[1].item() + 1 * probs[2].item()
        sent_scores.append(score)
        sent_neg.append(probs[0].item())
        sent_neu.append(probs[1].item())
        sent_pos.append(probs[2].item())
        sent_labels.append(["negative", "neutral", "positive"][probs.argmax().item()])

# ── Build results table ──
df = pd.DataFrame({
    "tweet": tweets,
    "topic": topics,
    "expected_lean": expected_lean,
    "expected_emotion": expected_emotion,
    "stance_score": stance_scores,
    "P(conservative)": cons_raw,
    "P(progressive)": prog_raw,
    "sentiment_score": sent_scores,
    "sent_label": sent_labels,
    "P(neg)": sent_neg,
    "P(neu)": sent_neu,
    "P(pos)": sent_pos,
})

# ── Analysis ──
print("\n" + "=" * 70)
print("RESULTS: STANCE MODEL")
print("=" * 70)

print("\n--- Stance by expected political lean ---")
stance_by_lean = df.groupby("expected_lean")["stance_score"].agg(["mean", "std", "count"])
print(stance_by_lean.round(4))

print("\n--- Stance by topic ---")
stance_by_topic = df.groupby("topic")["stance_score"].agg(["mean", "std"]).round(4)
print(stance_by_topic)

print("\n--- Accuracy: does stance sign match expected lean? ---")
df["stance_predicted"] = df["stance_score"].apply(
    lambda x: "conservative" if x > 0.05 else ("progressive" if x < -0.05 else "neutral"))
correct = (df["expected_lean"] == df["stance_predicted"]).sum()
print(f"Exact match: {correct}/{len(df)} ({correct/len(df)*100:.1f}%)")

# For non-neutral expected tweets only
non_neutral = df[df["expected_lean"] != "neutral"]
direction_correct = ((non_neutral["expected_lean"] == "conservative") &
                      (non_neutral["stance_score"] > 0)) | \
                     ((non_neutral["expected_lean"] == "progressive") &
                      (non_neutral["stance_score"] < 0))
print(f"Direction correct (non-neutral only): {direction_correct.sum()}/{len(non_neutral)} "
      f"({direction_correct.sum()/len(non_neutral)*100:.1f}%)")

print("\n" + "=" * 70)
print("RESULTS: SENTIMENT MODEL")
print("=" * 70)

print("\n--- Sentiment by expected emotion ---")
sent_by_emotion = df.groupby("expected_emotion")["sentiment_score"].agg(["mean", "std", "count"])
print(sent_by_emotion.round(4))

print("\n--- Sentiment by expected lean ---")
sent_by_lean = df.groupby("expected_lean")["sentiment_score"].agg(["mean", "std"])
print(sent_by_lean.round(4))

print("\n--- Sentiment by topic ---")
sent_by_topic = df.groupby("topic")["sentiment_score"].agg(["mean", "std"]).round(4)
print(sent_by_topic)

print("\n" + "=" * 70)
print("KEY COMPARISON: Democracy_Law tweets")
print("=" * 70)

dem_law = df[df["topic"].str.startswith("DemLaw")]
print(f"\n{'Tweet':<75s} {'Expected':<14s} {'Stance':>7s} {'Sent':>7s}")
print("-" * 110)
for _, row in dem_law.iterrows():
    tweet_short = row["tweet"][:72] + "..." if len(row["tweet"]) > 72 else row["tweet"]
    print(f"{tweet_short:<75s} {row['expected_lean']:<14s} {row['stance_score']:>+.3f} {row['sentiment_score']:>+.3f}")

print("\n" + "=" * 70)
print("PROBLEMATIC CASES: where stance gets direction wrong")
print("=" * 70)

wrong = non_neutral[~direction_correct]
print(f"\n{len(wrong)} misclassified tweets:\n")
for _, row in wrong.iterrows():
    tweet_short = row["tweet"][:65] + "..." if len(row["tweet"]) > 65 else row["tweet"]
    print(f"  Expected: {row['expected_lean']:<14s} Got stance: {row['stance_score']:>+.3f}  | {tweet_short}")

print("\n" + "=" * 70)
print("CROSS-ANALYSIS: Stance vs Sentiment correlation")
print("=" * 70)

from scipy import stats
r_overall, p_overall = stats.pearsonr(df["stance_score"], df["sentiment_score"])
print(f"\nOverall correlation: r = {r_overall:+.3f}, p = {p_overall:.4f}")
print("(Low correlation = they measure different things = good)")

for topic in sorted(df["topic"].unique()):
    sub = df[df["topic"] == topic]
    if len(sub) >= 5:
        r, p = stats.pearsonr(sub["stance_score"], sub["sentiment_score"])
        print(f"  {topic:<30s}  r = {r:+.3f}  p = {p:.3f}")

# ── Save full results ──
df.to_csv("stance_vs_sentiment_test_results.csv", index=False)
print(f"\nFull results saved to stance_vs_sentiment_test_results.csv")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
STANCE MODEL measures: political direction (conservative vs progressive)
  - Uses NLI entailment: 'does this tweet entail conservative/progressive views?'
  - Good at: clearly partisan statements
  - Bad at: factual statements, sarcasm, cross-cutting issues

SENTIMENT MODEL measures: emotional valence (positive vs negative)
  - Good at: capturing anger, enthusiasm, frustration
  - Bad at: distinguishing political direction

KEY QUESTION: Which is more useful for predicting voting behavior?
- If stance works well -> political lean of discourse predicts votes (but risk of tautology)
- If sentiment works -> emotional temperature predicts votes (more novel finding)
""")
