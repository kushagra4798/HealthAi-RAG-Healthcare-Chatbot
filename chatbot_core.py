import ollama
import numpy as np
import difflib
from knowledge_base import knowledge_base, hospital_info, faqs

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3"
BOT_NAME = "HealthAi"
CONFIDENCE_THRESHOLD = 0.55
NO_MATCH_THRESHOLD = 0.35

emergency_keywords = [
    "chest pain", "cant breathe", "can't breathe", "cannot breathe", "not breathing",
    "difficulty breathing", "shortness of breath", "unconscious", "unresponsive",
    "severe bleeding", "heavy bleeding", "seizure", "stroke", "slurred speech",
    "choking", "overdose", "poisoning", "severe burn", "heart attack", "cardiac arrest",
    "fainted", "collapsed",
]

self_harm_keywords = [
    "suicide", "kill myself", "end my life", "self harm", "selfharm", "hurt myself",
]

red_flag_rules = [
    (["yello", "eye"], "Gastroenterology"),
    (["yello", "skin"], "Gastroenterology"),
    (["jaundic"], "Gastroenterology"),
    (["blood", "stool"], "Gastroenterology"),
    (["blood", "vomit"], "Gastroenterology"),
    (["black", "stool"], "Gastroenterology"),
    (["tarry", "stool"], "Gastroenterology"),
    (["lump", "neck"], "Oncology"),
    (["lump", "grow"], "Oncology"),
    (["testicular"], "Urology"),
    (["blood", "urine"], "Urology"),
    (["bladder", "control"], "Orthopedics & Spine"),
    (["bowel", "control"], "Orthopedics & Spine"),
]


def find_red_flag_specialty(text):
    lowered = text.lower()
    for stems, specialty in red_flag_rules:
        if all(stem in lowered for stem in stems):
            return specialty
    return None


FUZZY_THRESHOLD = 0.8
EMERGENCY_FUZZY_THRESHOLD = 0.75


def fuzzy_word_in(word, text_words, threshold=FUZZY_THRESHOLD):
    for w in text_words:
        if difflib.SequenceMatcher(None, word, w).ratio() >= threshold:
            return True
    return False


def fuzzy_phrase_in(phrase, text_words, threshold=FUZZY_THRESHOLD):
    phrase_words = phrase.split()
    for pw in phrase_words:
        if not fuzzy_word_in(pw, text_words, threshold):
            return False
    return True


def get_embedding(text, task):
    prefixed_text = task + ": " + text
    response = ollama.embeddings(model=EMBED_MODEL, prompt=prefixed_text)
    return np.array(response["embedding"])


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_index():
    index = []
    for item in knowledge_base:
        text = item["specialty"] + ". " + item["symptoms"]
        embedding = get_embedding(text, "search_document")
        index.append({"embedding": embedding, "data": item})
    return index


def search(query, index, top_k=3):
    query_embedding = get_embedding(query, "search_query")
    scored = []
    for entry in index:
        score = cosine_similarity(query_embedding, entry["embedding"])
        scored.append((score, entry["data"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def format_doctors(doctors):
    lines = []
    for doc in doctors:
        lines.append(doc["name"] + " (" + doc["qualification"] + ", " + doc["experience"] + " experience)")
    return ", ".join(lines)


def check_emergency(text):
    words = text.lower().split()
    for phrase in self_harm_keywords:
        if fuzzy_phrase_in(phrase, words, EMERGENCY_FUZZY_THRESHOLD):
            return (
                "I'm really glad you reached out, and I want you to know you don't have to go through this alone. "
                "It would help to talk to a mental health professional or a crisis helpline as soon as you can. "
                "In India, the KIRAN mental health helpline is available 24x7 at 1800-599-0019. "
                "If you're in immediate danger, please head to the nearest emergency room right away."
            )
    for phrase in emergency_keywords:
        if fuzzy_phrase_in(phrase, words, EMERGENCY_FUZZY_THRESHOLD):
            return (
                "That sounds serious, and I don't want you to wait on this. "
                "Please get to the nearest emergency room or call emergency services right now. "
                "You can also reach us at " + hospital_info["name"] + " on " + hospital_info["phone"] + " if that helps."
            )
    return None


def check_faq(text):
    lowered = text.lower()
    for faq in faqs:
        for keyword in faq["keywords"]:
            if keyword in lowered:
                return faq["answer"]
    return None


specialty_gaps = [
    {
        "name": "Dermatology",
        "keywords": [
            "rash", "itch", "spots", "eczema", "psoriasis", "hives", "acne", "mole",
            "skin peeling", "dandruff", "skin lesion", "pimple",
        ],
        "suggested_doctor": "Dr. Mohd Shahid (Internal Medicine)",
    },
    {
        "name": "Ophthalmology",
        "keywords": [
            "blurred vision", "vision loss", "eye pain", "red eye", "watery eyes",
            "double vision", "difficulty seeing", "eye infection", "cataract", "glaucoma",
        ],
        "suggested_doctor": "Dr. Mohd Shahid (Internal Medicine)",
    },
    {
        "name": "Nephrology",
        "keywords": ["dialysis", "kidney failure", "chronic kidney disease", "ckd"],
        "suggested_doctor": "Dr. Vineet Singh Somvanshi (Urology)",
    },
    {
        "name": "Rheumatology",
        "keywords": ["rheumatoid arthritis", "lupus", "autoimmune disease"],
        "suggested_doctor": "Dr. Rahul Rishi (Orthopedics)",
    },
    {
        "name": "Allergy and Immunology",
        "keywords": ["allergic reaction", "hay fever", "anaphylaxis", "food allergy"],
        "suggested_doctor": "Dr. Mohd Shahid (Internal Medicine)",
    },
    {
        "name": "Endocrinology",
        "keywords": ["thyroid problem", "thyroid issue", "hormone imbalance", "hypothyroid", "hyperthyroid"],
        "suggested_doctor": "Dr. Mohd Shahid (Internal Medicine)",
    },
]


def check_specialty_gap(text):
    words = text.lower().split()
    for gap in specialty_gaps:
        for keyword in gap["keywords"]:
            if fuzzy_phrase_in(keyword, words):
                return (
                    "That sounds like it might need a " + gap["name"] + " specialist, but I don't see one listed "
                    "in our current directory for this hospital. It's best to call " + hospital_info["phone"] + " "
                    "to check availability, or " + gap["suggested_doctor"] + " can take an initial look and "
                    "guide you from there."
                )
    return None


def build_context_prompt(results):
    context = ""
    valid_specialties = []
    for score, data in results:
        context += "Specialty: " + data["specialty"] + "\n"
        context += "Symptoms: " + data["symptoms"] + "\n"
        context += "Doctors: " + format_doctors(data["doctors"]) + "\n\n"
        valid_specialties.append(data["specialty"])

    return (
        "You are a warm, friendly healthcare assistant for " + hospital_info["name"] + ".\n"
        "You help people understand which specialty and doctor to consult based on their symptoms, "
        "using only the reference information below. You are not a replacement for a real doctor.\n\n"
        "Reference information:\n" + context +
        "Pick exactly one specialty from this list that best matches the symptoms: " + ", ".join(valid_specialties) + ".\n"
        "If the person mentions both an emotional symptom (like feeling low or stressed) and a physical symptom "
        "(like yellowing eyes, unusual pain, or a lump), the physical symptom is usually more important medically "
        "and should guide your specialty choice, even if the emotional wording is more prominent in their message.\n"
        "You cannot book appointments. If it seems relevant, tell them to call the hospital to book, "
        "but never offer to book one yourself.\n"
        "Reply in exactly this format, with nothing before or after it:\n"
        "SPECIALTY: <the specialty you picked>\n"
        "ANSWER: <a warm, conversational reply. Briefly acknowledge how the person is feeling, "
        "gently explain why this specialty fits their symptoms in a sentence or two, "
        "then mention the doctor name(s). Keep it natural and caring, not clinical, blunt, or overly casual, "
        "and concise, no more than 4-5 sentences.>"
    )


def parse_model_reply(raw_reply, valid_specialties, fallback_specialty):
    specialty = None
    answer = raw_reply

    if "SPECIALTY:" in raw_reply and "ANSWER:" in raw_reply:
        specialty_part = raw_reply.split("SPECIALTY:")[1].split("ANSWER:")[0].strip()
        answer_part = raw_reply.split("ANSWER:")[1].strip()
        if specialty_part:
            specialty = specialty_part
        if answer_part:
            answer = answer_part

    if not specialty:
        for candidate in valid_specialties:
            if candidate.lower() in raw_reply.lower():
                specialty = candidate
                break

    if not specialty:
        specialty = fallback_specialty

    return specialty, answer


def get_no_match_reply():
    return (
        "Hmm, I'm not fully confident which specialty fits best based on what you've shared so far. "
        "A good starting point would be our Internal Medicine doctor, who can take a closer look and guide you from there. "
        "Or if it's easier, feel free to call " + hospital_info["name"] + " at " + hospital_info["phone"] + " "
        "and describe things to the front desk directly."
    )


def get_greeting():
    return (
        "Hi there! 👋 I'm " + BOT_NAME + ", the virtual assistant for " + hospital_info["name"] + ". "
        "I'm here to help you figure out which doctor might be right for what you're experiencing, "
        "and I'm happy to answer a few basic questions about the hospital too. "
        "Just a heads up, I'm not a substitute for a real doctor, so if this is ever an emergency, "
        "please call " + hospital_info["phone"] + " or head to the nearest hospital straight away.\n\n"
        "So, what's going on today? Feel free to describe it in your own words."
    )


def ask_chatbot(query, index, history, followup_asked):
    emergency_reply = check_emergency(query)
    if emergency_reply:
        return emergency_reply, None, None, followup_asked

    faq_reply = check_faq(query)
    if faq_reply:
        return faq_reply, None, None, followup_asked

    gap_reply = check_specialty_gap(query)
    if gap_reply:
        return gap_reply, None, None, followup_asked

    past_user_messages = " ".join([h["content"] for h in history if h["role"] == "user"])
    combined_query = (past_user_messages + " " + query).strip()

    results = search(combined_query, index)
    top_score = float(results[0][0])

    flagged_specialty = find_red_flag_specialty(combined_query)
    if flagged_specialty:
        already_included = any(data["specialty"] == flagged_specialty for score, data in results)
        if not already_included:
            flagged_entry = next(item for item in knowledge_base if item["specialty"] == flagged_specialty)
            results = [(1.0, flagged_entry)] + results[:2]
        else:
            results.sort(key=lambda r: 0 if r[1]["specialty"] == flagged_specialty else 1)
        top_score = 1.0

    if top_score < NO_MATCH_THRESHOLD and followup_asked:
        return get_no_match_reply(), top_score, None, followup_asked

    if top_score < CONFIDENCE_THRESHOLD and not followup_asked:
        follow_up = (
            "Got it, thanks for sharing that. Could you tell me a little more so I can point you in the right direction? "
            "For example, how long you've been dealing with this, how it feels, and if there's anything else going on alongside it."
        )
        return follow_up, top_score, None, True

    system_prompt = build_context_prompt(results)
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": query}]
    response = ollama.chat(model=CHAT_MODEL, messages=messages)
    raw_reply = response["message"]["content"]

    fallback_specialty = results[0][1]["specialty"]
    valid_specialties = [data["specialty"] for score, data in results]
    specialty, answer = parse_model_reply(raw_reply, valid_specialties, fallback_specialty)

    return answer, top_score, specialty, followup_asked
