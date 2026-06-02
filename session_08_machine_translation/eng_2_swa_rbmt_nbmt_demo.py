"""
===============================================================================
COMPLETE MACHINE TRANSLATION SYSTEM: RULE-BASED vs NEURAL APPROACHES
===============================================================================
This script implements and compares two distinct machine translation approaches:

1. **Rule-Based Machine Translation (RBMT)**
   Translates text using hand-crafted bilingual dictionaries,
   morphological rules, and structural transfer heuristics.
   It is transparent, interpretable, and requires no training data,
   but it does not scale well to unseen vocabulary or complex syntax.

2. **Neural-Based Machine Translation (NBMT)**
   Uses the Helsinki-NLP Opus-MT pre-trained transformer model
   (MarianMT) to translate with learned contextual representations.
   This approach captures long-range dependencies and idiomatic
   expressions far better than rule-based systems.

Task:
  Translate English paragraphs (UK English) to Kiswahili
  Compare the output quality of both approaches using evaluation metrics

Author: Xamdi Salaad
Date: 26-05-2026
"""
#===============================================================================
# SECTION 0: IMPORTS AND CONFIGURATION
#===============================================================================

#import required modules
import warnings
import re
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path

#suppress warnings
warnings.filterwarnings("ignore")

# Import torch conditionally (will be needed for neural model)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not installed. Neural translation will not work.")

# Import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] Matplotlib not installed. Visualization will be disabled.")

#===============================================================================
# SECTION 1: SOURCE TEXT (TWO ENGLISH UK PARAGRAPHS)
#===============================================================================

SOURCE_PARAGRAPHS: list[str] = [
    (
        "Last autumn, I travelled by train from Manchester to Edinburgh to visit an old "
        "university friend whom I had not seen in several years. The journey began very "
        "early in the morning, and the station was already crowded with commuters carrying "
        "coffee cups and briefcases. Although the weather forecast had predicted heavy rain "
        "throughout the day, the skies remained surprisingly clear for most of the trip. "
        "I spent part of the journey reading a historical novel and occasionally looking "
        "out of the window at the changing countryside. As the train moved further north, "
        "the hills became steeper and the villages appeared smaller and quieter. A family "
        "sitting nearby discussed their holiday plans in great detail, which provided an "
        "unexpected source of entertainment during the long journey. By the time I finally "
        "arrived in Edinburgh, I felt both tired and excited about spending the weekend "
        "exploring the city once again."
    ),
    (
        "In December, I travelled by bus from Nairobi to Arusha to attend a regional "
        "business conference that brought together young entrepreneurs from across East "
        "Africa. The journey began before sunrise, and the roads were already busy with "
        "lorries transporting goods between Kenya and Tanzania. Although the border "
        "crossing took longer than expected because of increased security checks, most "
        "passengers remained patient and continued chatting throughout the delay. I spent "
        "much of the trip listening to music, replying to messages on my phone, and "
        "observing the changing landscape outside the window. As we travelled further "
        "south, I noticed large farms, roadside markets, and groups of cyclists moving "
        "between small towns. Several passengers discussed the rising cost of fuel and "
        "how it had affected transport prices in recent months. By the time we finally "
        "arrived in Arusha during the evening, the streets were lively with traders, "
        "tourists, and people preparing for the holiday season."
    ),
]

#===============================================================================
# SECTION 2: HUMAN/REFERENCE KISWAHILI TRANSLATION (USED FOR METRIC COMPUTATION)
#===============================================================================

REFERENCE_TRANSLATION: list[str] = [
    (
        "Mwisho wa msimu wa vuli uliopita, nilisafiri kwa treni kutoka Manchester hadi Edinburgh "
        "kutembelea rafiki wa chuo kikuu ambaye sikuwa nimemwona kwa miaka kadhaa. "
        "Safari ilianza sana asubuhi sana, na kituo kilikuwa tayari kimejaa wasafiri waliobeba "
        "vikombe vya kahawa na mikoba. Ingawa utabiri wa hali ya hewa ulitabiri mvua kubwa "
        "siku nzima, anga ilibaki wazi kwa kushangaza kwa sehemu kubwa ya safari. "
        "Nilitumia sehemu ya safari kusoma riwaya ya kihistoria na kuangalia mara kwa mara "
        "nje ya dirisha kwenye sehemu ya mashambani inayobadilika. Wakati gari-moshi liliposonga zaidi kaskazini, "
        "milima ikawa mwinuko na vijiji vilionekana kuwa vidogo na tulivu. Familia "
        "waliokaa karibu walijadili mipango yao ya likizo kwa undani sana, ambayo ilitoa "
        "chanzo kisichotarajiwa cha burudani wakati wa safari ndefu. Kwa wakati mimi hatimaye "
        "Nilipowasili Edinburgh, nilihisi uchovu na kufurahiya kutumia wikendi "
        "kuchunguza jiji kwa mara nyingine tena."
    ),
    (
        "Mwezi wa Desemba, nilisafiri kwa basi kutoka Nairobi kwenda Arusha kuhudhuria mkutano wa mkoa "
        "mkutano wa biashara uliowaleta pamoja wajasiriamali vijana kutoka Mashariki "
        "Afrika. Safari ilianza kabla ya jua kuchomoza, na barabara tayari zilikuwa na shughuli nyingi "
        "malori yanayosafirisha mizigo kati ya Kenya na Tanzania. Ingawa mpaka "
        "kuvuka kulichukua muda mrefu kuliko ilivyotarajiwa kwa sababu ya ukaguzi wa usalama ulioongezeka, wengi "
        "abiria walibaki wavumilivu na kuendelea kuzungumza wakati wote wa kuchelewa. Nilitumia "
        "safari nyingi za kusikiliza muziki, kujibu ujumbe kwenye simu yangu, na "
        "tukiangalia mabadiliko ya mazingira nje ya dirisha. Tuliposafiri zaidi "
        "Kusini, niliona mashamba makubwa, masoko ya kando ya barabara, na vikundi vya waendesha baiskeli wakitembea "
        "kati ya miji midogo. Abiria kadhaa walijadili kupanda kwa gharama ya mafuta na "
        "jinsi ilivyoathiri bei za usafiri katika miezi ya hivi karibuni. Kufikia wakati sisi hatimaye "
        "Nilifika Arusha jioni, mitaa ilikuwa imechangamka na wafanyabiashara,"
        "watalii, na watu wanaojiandaa kwa msimu wa likizo."
    )
]

#===============================================================================
# SECTION 3: RULE-BASED MACHINE TRANSLATION (RBMT) CLASS
#===============================================================================

class RuleBasedTranslator:
    """
    A rule-based machine translation system from English to Kiswahili.

    This translator uses:
    1. A bilingual lexicon mapping English words/phrases to Kiswahili
    2. Structural transfer rules for grammatical transformations
    3. Sentence segmentation and punctuation handling

    The approach is transparent and interpretable but limited to
    vocabulary and patterns defined in its rules.
    """

    # Bilingual lexicon: English lemma -> Kiswahili equivalent organised thematically for readability
    LEXICON: Dict[str, str] = {

    # --- Pronouns & determiners ---
    "i": "mimi", "my": "yangu", "we": "sisi", "our": "yetu",
    "the": "", "a": "", "an": "", "their": "yao", "its": "yake",
    "this": "hii", "that": "ile", "these": "hizi", "those": "zile",
    "which": "ambacho", "whom": "ambaye", "who": "ambaye",

    # --- Common verbs ---
    "travelled": "nilisafiri", "travel": "safiri",
    "visit": "kutembelea", "visited": "alitembelea",
    "began": "ilianza", "begin": "anza",
    "was": "ilikuwa", "were": "zilikuwa", "is": "ni", "are": "ni",
    "had": "alikuwa na", "have": "kuwa na",
    "predicted": "ilitabiri", "remained": "iliendelea kuwa",
    "spent": "nilitumia", "spend": "tumia",
    "reading": "nikisoma", "read": "soma",
    "looking": "nikitazama", "look": "tazama",
    "moved": "ilisogea", "move": "songea",
    "became": "ilizidi kuwa", "appear": "kuonekana",
    "appeared": "vilionekana", "sitting": "iliyokaa",
    "discussed": "walijadiliana", "discuss": "jadiliana",
    "provided": "ilitoa", "provide": "toa",
    "arrived": "wasili", "arrive": "fika",
    "felt": "nilihisi", "feel": "hisi",
    "exploring": "nikichunguza", "explore": "chunguza",
    "seen": "kuona", "see": "ona",
    "carried": "wakibeba", "carry": "beba", "carrying": "wakibeba",
    "attend": "kuhudhuria", "attended": "alihudhuria",
    "brought": "ilikutanisha", "bring": "kuleta",
    "transporting": "yanayobeba", "transport": "beba",
    "crossing": "kuvuka", "cross": "vuka",
    "took": "ilichukua", "take": "chukua",
    "chatting": "kuzungumza", "chat": "zungumza",
    "listening": "nikisikiliza", "listen": "sikiliza",
    "replying": "kujibu", "reply": "jibu",
    "observing": "kuangalia", "observe": "angalia",
    "noticed": "niliangalia", "notice": "angalia",
    "moving": "wakisogea", "preparing": "wakijiandaa",

    # --- Nouns – travel & transport ---
    "train": "treni", "bus": "basi", "lorries": "malori",
    "station": "stesheni", "border": "mpaka", "road": "barabara",
    "roads": "barabara", "journey": "safari", "trip": "safari",
    "ticket": "tiketi", "platform": "jukwaa",
    "cyclists": "waendesha baiskeli",

    # --- Nouns – people ---
    "commuters": "wasafiri", "passengers": "abiria",
    "family": "familia", "friend": "rafiki",
    "entrepreneurs": "wajasiriamali", "traders": "wasusi",
    "tourists": "watalii", "people": "watu",

    # --- Nouns – places & geography ---
    "manchester": "Manchester", "edinburgh": "Edinburgh",
    "nairobi": "Nairobi", "arusha": "Arusha",
    "kenya": "Kenya", "tanzania": "Tanzania",
    "africa": "Afrika", "east": "Mashariki",
    "city": "mji", "town": "mji", "towns": "miji",
    "village": "kijiji", "villages": "vijiji",
    "countryside": "mashambani", "hills": "milima",
    "landscape": "mazingira", "window": "dirisha",
    "streets": "mitaa", "farms": "mashamba",
    "markets": "masoko",

    # --- Nouns – objects ---
    "coffee": "kahawa", "cups": "vikombe",
    "briefcases": "mikoba ya ofisi", "novel": "riwaya",
    "music": "muziki", "phone": "simu",
    "messages": "ujumbe", "goods": "bidhaa",
    "fuel": "mafuta",

    # --- Nouns – time ---
    "morning": "asubuhi", "evening": "jioni",
    "weekend": "wikendi", "day": "siku",
    "months": "miezi", "years": "miaka",
    "autumn": "vuli", "december": "Desemba",
    "sunrise": "mapambazuko",

    # --- Adjectives & adverbs ---
    "old": "wa zamani", "historical": "ya kihistoria",
    "heavy": "kubwa", "clear": "wazi",
    "early": "mapema", "further": "zaidi",
    "north": "kaskazini", "south": "kusini",
    "small": "ndogo", "smaller": "vidogo",
    "large": "kubwa", "great": "mkubwa",
    "long": "ndefu", "short": "fupi",
    "tired": "uchovu", "excited": "msisimko",
    "young": "wachanga", "regional": "wa kikanda",
    "unexpected": "kisichotarajiwa",
    "surprisingly": "kwa mshangao",
    "patient": "subira", "lively": "na shughuli",
    "busy": "na shughuli nyingi", "already": "tayari",
    "crowded": "imejaa", "quiet": "kimya",
    "quieter": "kimya zaidi", "steep": "mirefu",
    "steeper": "mirefu zaidi",

    # --- Prepositions & conjunctions ---
    "from": "kutoka", "to": "hadi", "by": "kwa",
    "in": "katika", "at": "kwenye", "of": "ya",
    "for": "kwa", "with": "na", "about": "kuhusu",
    "during": "wakati wa", "throughout": "siku nzima",
    "between": "kati ya", "across": "kutoka",
    "because": "kwa sababu", "although": "ingawa",
    "as": "kadri", "when": "nilipofika",
    "out": "nje", "nearby": "karibu",

    # --- Others ---
    "not": "si", "both": "zote mbili", "finally": "hatimaye",
    "most": "wengi", "several": "kadhaa", "many": "nyingi",
    "much": "nyingi", "part": "sehemu", "time": "wakati",
    "once": "tena", "again": "tena", "very": "sana",
    "too": "pia", "also": "pia",
    "detail": "undani mkubwa", "plans": "mipango",
    "holiday": "likizo", "season": "msimu",
    "university": "chuo", "conference": "mkutano",
    "business": "biashara", "security": "usalama",
    "checks": "ukaguzi", "delay": "kuchelewa",
    "cost": "gharama", "prices": "bei",
    "entertainment": "burudani", "source": "chanzo",
    "rain": "mvua", "weather": "hali ya hewa",
    "forecast": "utabiri", "skies": "anga"
    }

    # Structural transfer rules: (regex_pattern, replacement_string), applied in order after lexicon substitution
    TRANSFER_RULES: List[Tuple[str, str]] = [
        # Negation: "si ilikuwa" -> retain negation marker
        (r'\bsi ilikuwa\b', "haikuwa"),
        # Possessive smoothing
        (r'rafiki yangu wa zamani wa chuo', 'rafiki yangu wa zamani wa chuo'),
        # Remove orphaned articles left by empty-string lexicon entries
        (r'\s{2,}', " "),
        # Fix spacing after punctuation
        (r'\s+([.!?,;:])', r'\1'),
        # Capitalize first letter of sentences
        (r'(^|\.\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper())
    ]

    def __init__(self) -> None:
        """Initialize the translator with sorted lexicon for longest-match first strategy."""
        self._sorted_lexicon: List[Tuple[str, str]] = sorted(
            self.LEXICON.items(), key=lambda kv: len(kv[0]), reverse=True
        )

    #----------------------------------------------------------------------------
    # I. Public Interface
    #----------------------------------------------------------------------------
    def translate(self, text: str) -> str:
        """
        Translate English text to Kiswahili using rule-based approach.

        Args:
            text: English input text (can be multiple sentences)

        Returns:
            Kiswahili translation
        """
        sentences: List[str] = self._split_sentences(text)
        translated: List[str] = [self._translate_sentence(s) for s in sentences]
        return " ".join(translated)

    #----------------------------------------------------------------------------
    # II. Private helpers
    #----------------------------------------------------------------------------
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Split text into sentences using punctuation markers.

        Args:
            text: Input text

        Returns:
            List of individual sentences
        """
        raw: List[str] = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    def _translate_sentence(self, sentence: str) -> str:
        """
        Translate a single sentence using lexicon lookup and transfer rules.

        Args:
            sentence: English sentence to translate

        Returns:
            Kiswahili translation of the sentence
        """
        #------------------ Stage 1: Analysis---------------
        # Preserving trailing punctuation
        punct: str = ""
        if sentence and sentence[-1] in ".!?;:":
            punct = sentence[-1]
            sentence = sentence[:-1]

        tokens: List[str] = sentence.lower().split()

        #------------------ Stage 2a: Lexicon transfer---------------
        output_tokens: List[str] = []
        idx: int = 0
        while idx < len(tokens):
            matched: bool = False
            # Try multi-word matching first (up to 3 tokens)
            for window in (3, 2, 1):
                if idx + window <= len(tokens):
                    phrase: str = " ".join(tokens[idx:idx+window])
                    if phrase in self.LEXICON:
                        replacement: str = self.LEXICON[phrase]
                        if replacement:  # Skip empty-string mappings (articles)
                            output_tokens.append(replacement)
                            idx += window
                            matched = True
                            break
            if not matched:
                # Unknown token: pass through as-is
                tok: str = tokens[idx]
                output_tokens.append(tok)
                idx += 1

        output: str = " ".join(output_tokens)

        #------------------ Stage 2b: Structural transfer rules---------------
        for pattern, replacement in self.TRANSFER_RULES:
            output = re.sub(pattern, replacement, output, flags=re.IGNORECASE)

        #------------------ Stage 3: Generation---------------
        output = re.sub(r"\s{2,}", " ", output.strip())

        # Add back the punctuation
        if punct:
            output = output.rstrip() + punct

        # Capitalize first letter
        if output:
            output = output[0].upper() + output[1:]

        return output

#===============================================================================
# SECTION 4: NEURAL-BASED MACHINE TRANSLATION (NBMT) CLASS
#===============================================================================

class NeuralTranslator:
    """
    A neural machine translation system using Helsinki-NLP's Opus-MT model.

    This translator uses a pre-trained transformer model (MarianMT) to perform
    context-aware translation from English to Kiswahili. Unlike rule-based systems,
    it learns patterns from millions of parallel texts and can handle:
    - Idiomatic expressions
    - Long-range dependencies
    - Unseen vocabulary through subword tokenization
    - Complex grammatical structures

    The model is loaded lazily (only when first used) to save memory if not needed.
    """

    DEFAULT_MODEL: str = "Helsinki-NLP/opus-mt-en-swc"

    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 512, beam_size: int = 4) -> None:
        """
        Initialize the neural translator with model configuration.

        Args:
            model_name: Hugging Face model identifier for the translation model
            max_length: Maximum token length for input sequences
            beam_size: Beam size for beam search decoding (higher = better but slower)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.beam_size = beam_size
        self._pipeline = None  # Lazy-loaded on first call
        self._tokenizer = None
        self._model = None

    def _load_model(self) -> None:
        """
        Load the MarianMT model and tokenizer from Hugging Face.
        This is called automatically on the first translation request.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural translation. Please install: pip install torch")

        try:
            from transformers import MarianMTModel, MarianTokenizer

            print(f"\n[NBMT] Loading model '{self.model_name}'...")
            print("[NBMT] This may take a moment on first run as the model is downloaded...")

            # Load tokenizer and model
            self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self._model = MarianMTModel.from_pretrained(self.model_name)

            # Set to evaluation mode
            self._model.eval()

            print("[NBMT] Model loaded successfully!")
        except ImportError:
            raise ImportError("Transformers library required. Please install: pip install transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load neural translation model: {e}")

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into smaller chunks to avoid exceeding token limits.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks within token limits
        """
        # Simple sentence-based chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # Rough estimate: 1.3 tokens per word on average
            approx_tokens = len(sentence.split()) * 1.3
            if current_length + approx_tokens > self.max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = approx_tokens
                else:
                    chunks.append(sentence)
                    current_length = 0
            else:
                current_chunk.append(sentence)
                current_length += approx_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def translate(self, text: str) -> str:
        """
        Translate English text to Kiswahili using the neural model.

        Args:
            text: English input text (can be multiple sentences/paragraphs)

        Returns:
            Kiswahili translation
        """
        # Lazy load the model on first use
        if self._model is None:
            self._load_model()

        # Split into paragraphs/chunks to avoid exceeding token limits
        chunks: List[str] = self._chunk_text(text)
        translations: List[str] = []

        print(f"[NBMT] Translating {len(chunks)} chunk(s)...")

        for i, chunk in enumerate(chunks, 1):
            try:
                # Tokenize input
                encoded = self._tokenizer(
                    [chunk],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )

                # Generate translation
                with torch.no_grad():
                    generated_ids = self._model.generate(
                        **encoded,
                        num_beams=self.beam_size,
                        max_length=self.max_length,
                        early_stopping=True,
                        no_repeat_ngram_size=3
                    )

                # Decode the generated tokens
                decoded = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                translations.append(decoded)

                print(f"[NBMT] Chunk {i}/{len(chunks)} translated successfully")

            except Exception as e:
                print(f"[NBMT] Error translating chunk {i}: {e}")
                translations.append(f"[TRANSLATION ERROR: {e}]")

        # Join all chunks with spaces
        return " ".join(translations)

#===============================================================================
# SECTION 5: EVALUATION METRICS
#===============================================================================

class TranslationEvaluator:
    """
    Evaluate translation quality using standard metrics.

    Metrics implemented:
    - BLEU (Bilingual Evaluation Understudy): Measures n-gram overlap
    - ChrF (Character n-gram F-score): Character-based evaluation
    - ROUGE-L: Longest common subsequence overlap
    """

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Simple tokenization for evaluation metrics.

        Args:
            text: Input text string

        Returns:
            List of lowercase tokens
        """
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", " ", text)
        return text.split()

    @staticmethod
    def bleu_score(reference: str, hypothesis: str, n: int = 4) -> float:
        """
        Calculate BLEU score (simplified version).

        Args:
            reference: Reference (ground truth) translation
            hypothesis: Machine-generated translation
            n: Maximum n-gram order

        Returns:
            BLEU score between 0 and 1
        """
        ref_tokens = TranslationEvaluator.tokenize(reference)
        hyp_tokens = TranslationEvaluator.tokenize(hypothesis)

        # Calculate precision for each n-gram order
        precisions = []
        for i in range(1, n + 1):
            ref_ngrams = TranslationEvaluator._get_ngrams(ref_tokens, i)
            hyp_ngrams = TranslationEvaluator._get_ngrams(hyp_tokens, i)

            if not hyp_ngrams:
                precisions.append(0)
                continue

            # Count matches (with clipping)
            match_count = 0
            ref_ngrams_copy = ref_ngrams.copy()
            for ngram in hyp_ngrams:
                if ngram in ref_ngrams_copy:
                    match_count += 1
                    # Remove to avoid overcounting (clipping)
                    ref_ngrams_copy.remove(ngram)

            precision = match_count / len(hyp_ngrams) if hyp_ngrams else 0
            precisions.append(precision)

        # Calculate brevity penalty
        if len(hyp_tokens) > len(ref_tokens):
            bp = 1.0
        else:
            bp = min(1.0, pow(2.71828, 1 - len(ref_tokens) / max(1, len(hyp_tokens))))

        # Calculate geometric mean of precisions
        import math
        if min(precisions) > 0:
            geometric_mean = pow(math.prod(precisions), 1.0 / n)
        else:
            geometric_mean = 0

        return bp * geometric_mean

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> List[tuple]:
        """
        Generate n-grams from token list.

        Args:
            tokens: List of tokens
            n: n-gram order

        Returns:
            List of n-gram tuples
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def rouge_l(reference: str, hypothesis: str) -> float:
        """
        Calculate ROUGE-L score based on longest common subsequence.

        Args:
            reference: Reference translation
            hypothesis: Machine-generated translation

        Returns:
            ROUGE-L F1 score between 0 and 1
        """
        ref_tokens = TranslationEvaluator.tokenize(reference)
        hyp_tokens = TranslationEvaluator.tokenize(hypothesis)

        # Find LCS length using dynamic programming
        lcs_length = TranslationEvaluator._lcs_length(ref_tokens, hyp_tokens)

        if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            return 0.0

        recall = lcs_length / len(ref_tokens)
        precision = lcs_length / len(hyp_tokens)

        if precision + recall == 0:
            return 0.0

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    @staticmethod
    def _lcs_length(a: List[str], b: List[str]) -> int:
        """
        Calculate length of longest common subsequence.

        Args:
            a: First sequence
            b: Second sequence

        Returns:
            Length of LCS
        """
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    @staticmethod
    def evaluate_all(reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate all available evaluation metrics.

        Args:
            reference: Reference translation
            hypothesis: Machine-generated translation

        Returns:
            Dictionary containing BLEU and ROUGE-L scores
        """
        return {
            'BLEU-4': TranslationEvaluator.bleu_score(reference, hypothesis, n=4),
            'ROUGE-L': TranslationEvaluator.rouge_l(reference, hypothesis)
        }

    def _print_scores(self, system_name: str, scores: Dict[str, float]) -> None:
        """
        Print formatted evaluation scores.

        Args:
            system_name: Name of the translation system
            scores: Dictionary of metric scores
        """
        print(f"\n{system_name}:")
        print("  " + "-" * 30)
        for metric, score in scores.items():
            print(f"  {metric:12}: {score:.4f}")

    def visualize_comparison(self, rbmt_scores_list: List[Dict], nbmt_scores_list: List[Dict]) -> None:
        """
        Create visualization comparing RBMT and NBMT performance.

        Args:
            rbmt_scores_list: List of score dictionaries for RBMT across paragraphs
            nbmt_scores_list: List of score dictionaries for NBMT across paragraphs
        """
        if not MATPLOTLIB_AVAILABLE:
            print("[WARNING] Matplotlib not available. Skipping visualization.")
            return

        # Calculate average scores
        rbmt_avg = self.avg_scores(rbmt_scores_list)
        nbmt_avg = self.avg_scores(nbmt_scores_list)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Machine Translation System Comparison: RBMT vs NBMT', fontsize=16, fontweight='bold')

        # Plot 1: Bar chart comparison
        ax1 = axes[0]
        metrics = list(rbmt_avg.keys())
        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax1.bar(x - width/2, [rbmt_avg[m] for m in metrics], width, label='RBMT', color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, [nbmt_avg[m] for m in metrics], width, label='NBMT', color='lightcoral', alpha=0.8)

        ax1.set_xlabel('Evaluation Metrics', fontsize=12)
        ax1.set_ylabel('Scores (0-1 scale)', fontsize=12)
        ax1.set_title('Average Performance Comparison', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        # Plot 2: Line chart showing per-paragraph performance
        ax2 = axes[1]
        paragraphs = [f'Para {i+1}' for i in range(len(rbmt_scores_list))]

        # Extract BLEU scores per paragraph
        rbmt_bleu = [s['BLEU-4'] for s in rbmt_scores_list]
        nbmt_bleu = [s['BLEU-4'] for s in nbmt_scores_list]

        ax2.plot(paragraphs, rbmt_bleu, 'o-', label='RBMT BLEU-4', color='skyblue', linewidth=2, markersize=8)
        ax2.plot(paragraphs, nbmt_bleu, 's-', label='NBMT BLEU-4', color='lightcoral', linewidth=2, markersize=8)

        ax2.set_xlabel('Paragraphs', fontsize=12)
        ax2.set_ylabel('BLEU-4 Score', fontsize=12)
        ax2.set_title('Per-Paragraph BLEU Score Comparison', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels on points
        for i, (rb, nb) in enumerate(zip(rbmt_bleu, nbmt_bleu)):
            ax2.annotate(f'{rb:.3f}', (paragraphs[i], rb), xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax2.annotate(f'{nb:.3f}', (paragraphs[i], nb), xytext=(5, -10), textcoords='offset points', fontsize=8)

        plt.tight_layout()
        plt.show()

        # Create additional visualization: Radar chart for comprehensive comparison
        fig2, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # Prepare data for radar chart
        categories = metrics
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        rbmt_values = [rbmt_avg[m] for m in metrics]
        nbmt_values = [nbmt_avg[m] for m in metrics]
        rbmt_values += rbmt_values[:1]  # Close the loop
        nbmt_values += nbmt_values[:1]  # Close the loop

        # Plot
        ax_radar.plot(angles, rbmt_values, 'o-', linewidth=2, label='RBMT', color='skyblue')
        ax_radar.fill(angles, rbmt_values, alpha=0.25, color='skyblue')
        ax_radar.plot(angles, nbmt_values, 'o-', linewidth=2, label='NBMT', color='lightcoral')
        ax_radar.fill(angles, nbmt_values, alpha=0.25, color='lightcoral')

        # Set category labels
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=11)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Multi-Metric Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        ax_radar.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def avg_scores(score_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate average scores across multiple evaluation results.

        Args:
            score_list: List of score dictionaries from multiple evaluations

        Returns:
            Dictionary with average scores for each metric
        """
        if not score_list:
            return {}

        keys: list[str] = list(score_list[0].keys())
        return {k: round(sum(s[k] for s in score_list) / len(score_list), 4) for k in keys}

#===============================================================================
# SECTION 6: MAIN EXECUTION AND COMPARISON
#===============================================================================

def main() -> None:
    """
    Main function to compare RBMT and NBMT approaches with visualization.
    """
    print("=" * 80)
    print("MACHINE TRANSLATION SYSTEM COMPARISON: RBMT vs NBMT")
    print("=" * 80)

    # Initialize translators
    print("\n[INIT] Initializing Rule-Based Translator...")
    rbmt = RuleBasedTranslator()

    print("[INIT] Initializing Neural Translator...")
    nbmt = NeuralTranslator()

    # Initialize evaluator
    evaluator = TranslationEvaluator()

    # Store results for comparison
    rbmt_scores_all = []
    nbmt_scores_all = []

    # Process each paragraph
    for i, source in enumerate(SOURCE_PARAGRAPHS, 1):
        print(f"\n{'=' * 80}")
        print(f"PARAGRAPH {i}")
        print(f"{'=' * 80}")
        print(f"\n[SOURCE TEXT (English)]:\n{source}\n")

        print(f"[REFERENCE TRANSLATION (Kiswahili)]:\n{REFERENCE_TRANSLATION[i-1]}\n")

        # RBMT Translation
        print("-" * 40)
        print("RULE-BASED TRANSLATION (RBMT):")
        print("-" * 40)
        try:
            rbmt_output = rbmt.translate(source)
            print(rbmt_output)

            # Evaluate RBMT
            rbmt_scores = evaluator.evaluate_all(REFERENCE_TRANSLATION[i-1], rbmt_output)
            rbmt_scores_all.append(rbmt_scores)
            print(f"\n[RBMT Scores] BLEU: {rbmt_scores['BLEU-4']:.4f}, ROUGE-L: {rbmt_scores['ROUGE-L']:.4f}")
        except Exception as e:
            print(f"[ERROR] RBMT failed: {e}")
            rbmt_scores_all.append({'BLEU-4': 0.0, 'ROUGE-L': 0.0})

        # NBMT Translation
        print("\n" + "-" * 40)
        print("NEURAL TRANSLATION (NBMT):")
        print("-" * 40)
        try:
            nbmt_output = nbmt.translate(source)
            print(nbmt_output)

            # Evaluate NBMT
            nbmt_scores = evaluator.evaluate_all(REFERENCE_TRANSLATION[i-1], nbmt_output)
            nbmt_scores_all.append(nbmt_scores)
            print(f"\n[NBMT Scores] BLEU: {nbmt_scores['BLEU-4']:.4f}, ROUGE-L: {nbmt_scores['ROUGE-L']:.4f}")
        except Exception as e:
            print(f"[ERROR] NBMT failed: {e}")
            nbmt_scores_all.append({'BLEU-4': 0.0, 'ROUGE-L': 0.0})

    # Calculate overall averages
    rbmt_avg: Dict[str, float] = TranslationEvaluator.avg_scores(rbmt_scores_all)
    nbmt_avg: Dict[str, float] = TranslationEvaluator.avg_scores(nbmt_scores_all)

    # Print overall evaluation
    print("\n\n" + " " * 78)
    print("OVERALL EVALUATION (Macro-average across all paragraphs)")
    print("\n" + " " * 78)
    evaluator._print_scores("RBMT - Overall Average", rbmt_avg)
    evaluator._print_scores("NBMT - Overall Average", nbmt_avg)

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<15} {'RBMT':<15} {'NBMT':<15} {'Improvement':<15}")
    print("-" * 60)
    for metric in rbmt_avg.keys():
        rbmt_score = rbmt_avg[metric]
        nbmt_score = nbmt_avg[metric]
        improvement = ((nbmt_score - rbmt_score) / rbmt_score * 100) if rbmt_score > 0 else float('inf')
        improvement_str = f"{improvement:+.1f}%" if improvement != float('inf') else "N/A"
        print(f"{metric:<15} {rbmt_score:<15.4f} {nbmt_score:<15.4f} {improvement_str:<15}")

    # Visualization
    if MATPLOTLIB_AVAILABLE:
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS...")
        print("=" * 80)
        evaluator.visualize_comparison(rbmt_scores_all, nbmt_scores_all)
    else:
        print("\n[INFO] Skipping visualization (matplotlib not installed)")
        print("To enable visualizations, install: pip install matplotlib numpy")

    # Final conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if nbmt_avg['BLEU-4'] > rbmt_avg['BLEU-4']:
        improvement = ((nbmt_avg['BLEU-4'] - rbmt_avg['BLEU-4']) / rbmt_avg['BLEU-4'] * 100) if rbmt_avg['BLEU-4'] > 0 else float('inf')
        print(f"\n✓ Neural Translation (NBMT) outperforms Rule-Based Translation by {improvement:.1f}% in BLEU score")
        print(f"  This demonstrates the advantage of learned representations over hand-crafted rules.")
        print(f"  NBMT better captures context, idioms, and grammatical structures.")
    elif nbmt_avg['BLEU-4'] == rbmt_avg['BLEU-4']:
        print(f"\n✓ Both systems achieved comparable performance on this test set.")
    else:
        print(f"\n✓ Rule-Based Translation (RBMT) performed better on this specific dataset.")
        print(f"  This may be due to limited vocabulary or domain-specific terminology.")

    print("\n[KEY OBSERVATIONS]")
    print("1. RBMT: Transparent and interpretable, zero training data required")
    print("   - Strengths: Fast, predictable, works offline")
    print("   - Weaknesses: Brittle, requires extensive lexicon maintenance")
    print("2. NBMT: Leverages learned representations from millions of examples")
    print("   - Strengths: Handles unseen words, better fluency, captures context")
    print("   - Weaknesses: Requires GPU for optimal performance, larger memory footprint")
    print("\n[RECOMMENDATION]")
    print("- Use RBMT for constrained environments with limited vocabulary")
    print("- Use NBMT for open-domain translation requiring natural fluency")

    print("\n" + "=" * 80)
    print("TRANSLATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()