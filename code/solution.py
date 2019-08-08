# ----------------------------------------------------------------------------------------
from contextlib import contextmanager
import datetime
import gc
import glob
# from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
import multiprocessing
import os
from os.path import join
import pickle
import random
import sys
import time

from fastprogress import master_bar, progress_bar
from joblib import Parallel, delayed
from keras.preprocessing import text, sequence
import matplotlib.pyplot as plt
from nltk.tokenize.treebank import TreebankWordTokenizer
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset

sys.path.append("../input/toxic-src")
sys.path.append("../input/gpt2-pytorch/pytorch-pretrained-bert-master/pytorch-pretrained-BERT-master")
# from logger import setup_logger, LOGGER
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, BertConfig
from utils_gpt import seed_torch, convert_line_fast, setting, trim_tensors, convert_lines_head_tail, \
    convert_lines_gpt2_head_tail

# from pytorch_pretrained_bert import GPT2Tokenizer, GPT2ClassificationHeadModel, OpenAIAdam, GPT2Config

# %matplotlib inline
# sns.set(style='ticks')
# tqdm.pandas()
# ----------------------------------------------------------------------------------------
"""
def get_logger():
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    # handler1
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(asctime)s %(levelname)8s %(message)s"))
    # handler2
    handler2 = FileHandler(filename=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".log")
    handler2.setFormatter(Formatter("%(asctime)s %(levelname)8s %(message)s"))
    # addHandler
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger
"""


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f'[{name}] done in {time.time() - t0:.0f} s')


def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def log_loss(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(np.log(y_pred) * y_true + np.log(1 - y_pred) * (1 - y_true))


# ----------------------------------------------------------------------------------------
# logger, config
# logger = get_logger()

# parameters
n_workers = 4
n_splits = 5
seed = 777
seed_everything(seed)

# parameters for RNN
maxlen = 300
max_features = 410047
batch_size = 512
lr = 0.001
epochs = 10

# path
CRAWL_EMBEDDING_PATH = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
GLOVE_EMBEDDING_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'
# GOOGLE_EMBEDDING_PATH = '../input/quoratextemb/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
# WIKI_EMBEDDING_PATH = '../input/quoratextemb/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

# constants
target = 'target'
aux_target = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
n_labels = len(aux_target) + 1

# text symbols
symbols_to_isolate = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'
symbols_to_delete = '\n🍕\r🐵😑\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
small_caps_mapping = {
    "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ғ": "f", "ɢ": "g", "ʜ": "h", "ɪ": "i",
    "ᴊ": "j", "ᴋ": "k", "ʟ": "l", "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ǫ": "q", "ʀ": "r",
    "s": "s", "ᴛ": "t", "ᴜ": "u", "ᴠ": "v", "ᴡ": "w", "x": "x", "ʏ": "y", "ᴢ": "z"
}
contraction_mapping = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
    "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
    "I've": "I have", "i'd": "i would", "i'd've":
        "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have", "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
    "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have",
    "there's": "there is",
    "here's": "here is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
    "when's": "when is",
    "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
    "who'll": "who will", "who'll've": "who will have",
    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
    "won't": "will not",
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
    "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
    "you're": "you are", "you've": "you have",
    "trump's": "trump is", "obama's": "obama is", "canada's": "canada is", "today's": "today is"
}
specail_signs = {"…": "...", "₂": "2"}
specials = ["’", "‘", "´", "`"]


# ----------------------------------------------------------------------------------------
class JigsawEvaluator:
    def __init__(self, y_true, y_identity, power=-5, overall_model_weight=0.25):
        self.y = (y_true >= 0.5).astype(int)
        self.y_i = (y_identity >= 0.5).astype(int)
        self.n_subgroups = self.y_i.shape[1]
        self.power = power
        self.overall_model_weight = overall_model_weight

    @staticmethod
    def _compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def _compute_subgroup_auc(self, i, y_pred):
        mask = self.y_i[:, i] == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bpsn_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bnsp_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y != 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def compute_bias_metrics_for_model(self, y_pred):
        records = np.zeros((3, self.n_subgroups))
        for i in range(self.n_subgroups):
            records[0, i] = self._compute_subgroup_auc(i, y_pred)
            records[1, i] = self._compute_bpsn_auc(i, y_pred)
            records[2, i] = self._compute_bnsp_auc(i, y_pred)
        return records

    def _calculate_overall_auc(self, y_pred):
        return roc_auc_score(self.y, y_pred)

    def _power_mean(self, array):
        total = sum(np.power(array, self.power))
        return np.power(total / len(array), 1 / self.power)

    def get_final_metric(self, y_pred):
        bias_metrics = self.compute_bias_metrics_for_model(y_pred)
        bias_score = np.average([
            self._power_mean(bias_metrics[0]),
            self._power_mean(bias_metrics[1]),
            self._power_mean(bias_metrics[2])
        ])
        overall_score = self.overall_model_weight * self._calculate_overall_auc(y_pred)
        bias_score = (1 - self.overall_model_weight) * bias_score
        return overall_score + bias_score, bias_metrics[0], bias_metrics[1], bias_metrics[2]


# ----------------------------------------------------------------------------------------
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
print(f'test shape: {test.shape}')
######################################## RNN Part ########################################
# ----------------------------------------------------------------------------------------
# ここを変える
# embedding_matrix = pd.read_pickle('../input/toxicpreprocesseddata/embedding_matrix.pkl')
# tokenizer = pd.read_pickle('../input/toxicpreprocesseddata/tokenizer.pkl')
# RNN text cleaning ----------------------------------------------------------------------
treebank_tokenizer = TreebankWordTokenizer()

isolate_dict = {ord(c): f' {c} ' for c in symbols_to_isolate}
remove_dict = {ord(c): f'' for c in symbols_to_delete}


def handle_punctuation(x):
    x = x.translate(remove_dict)
    x = x.translate(isolate_dict)
    return x


def handle_contractions(x):
    x = treebank_tokenizer.tokenize(x)
    return x


def fix_quote(x):
    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
    x = ' '.join(x)
    return x


def clean_text(x):
    x = handle_punctuation(x)
    x = handle_contractions(x)
    x = fix_quote(x)
    return x


def apply_clean_text(X):
    if not isinstance(X, pd.Series):
        X = pd.Series(X)
    return X.apply(clean_text)


def parallel_clean_text(X):
    with multiprocessing.Pool(processes=n_workers) as p:
        splits = np.array_split(X, n_workers)
        pool_results = p.map(apply_clean_text, splits)
    return np.concatenate(pool_results)


# ----------------------------------------------------------------------------------------
X_test = parallel_clean_text(test['comment_text'])


# ----------------------------------------------------------------------------------------
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path, 'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr


def build_embedding_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((max_features + 1, 300))
    unknown_words = []

    for word, i in word_index.items():
        if i <= max_features:
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                try:
                    embedding_matrix[i] = embedding_index[word.lower()]
                except KeyError:
                    try:
                        embedding_matrix[i] = embedding_index[word.title()]
                    except KeyError:
                        unknown_words.append(word)
    return embedding_matrix, unknown_words


# for stage2 -----------------------------------------------------------------------------
tokenizer = text.Tokenizer(num_words=max_features,
                           filters='',
                           lower=False)

tokenizer.fit_on_texts(list(X_test))
X_test = tokenizer.texts_to_sequences(X_test)
test_lengths = np.array([len(x) for x in X_test])
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
# for stage2 -----------------------------------------------------------------------------
crawl_matrix, unknown_words_crawl = build_embedding_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)
glove_matrix, unknown_words_glove = build_embedding_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)
del crawl_matrix, glove_matrix;
gc.collect()
max_features = max_features or len(tokenizer.word_index) + 1
print(f'Number of unknown words (crawl): {len(unknown_words_crawl)}')
print(f'Number of unknown words (glove): {len(unknown_words_glove)}')
print(f'max_features: {max_features}')


# tokenize -------------------------------------------------------------------------------
# X_test = tokenizer.texts_to_sequences(X_test)
# test_lengths = np.array([len(x) for x in X_test])

# all_tokens = []
# for tokens_a in X_test:
#    if len(tokens_a)>maxlen:
#        tokens_a = tokens_a[:150] + tokens_a[-150:]
#    all_tokens.append(tokens_a)

# X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
# X_test_head_tail = sequence.pad_sequences(all_tokens, maxlen=maxlen)
# dataloader -----------------------------------------------------------------------------
class SequenceBucketCollator():
    def __init__(self, choose_length, sequence_index, length_index, label_index=None):
        self.choose_length = choose_length
        self.sequence_index = sequence_index
        self.length_index = length_index
        self.label_index = label_index

    def __call__(self, batch):
        batch = [torch.stack(x) for x in list(zip(*batch))]

        sequences = batch[self.sequence_index]
        lengths = batch[self.length_index]

        length = self.choose_length(lengths)
        mask = torch.arange(start=maxlen, end=0, step=-1) < length
        padded_sequences = sequences[:, mask]

        batch[self.sequence_index] = padded_sequences

        if self.label_index is not None:
            return [x for i, x in enumerate(batch) if i != self.label_index], batch[self.label_index]

        return batch


def prepare_data_loader(X, lengths, y=None, shuffle=False):
    if y is None:
        dataset = TensorDataset(torch.from_numpy(X),
                                torch.from_numpy(lengths))
        collator = SequenceBucketCollator(lambda lenghts: lenghts.max(),
                                          sequence_index=0,
                                          length_index=1)
    else:
        dataset = TensorDataset(torch.from_numpy(X),
                                torch.from_numpy(lengths),
                                torch.tensor(y, dtype=torch.float32))
        collator = SequenceBucketCollator(lambda lenghts: lenghts.max(),
                                          sequence_index=0,
                                          length_index=1,
                                          label_index=2)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator, num_workers=n_workers)


# model ----------------------------------------------------------------------------------
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class EmbLSTM(nn.Module):
    def __init__(self, embedding_matrix, max_features, num_aux_targets=6):
        super().__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.4)

        self.lstm1 = nn.LSTM(embed_size, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(128 * 2, 128, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 512)

        self.linear_out = nn.Linear(512, 1)
        self.linear_aux_out = nn.Linear(512, num_aux_targets)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out


# RNN prediction -------------------------------------------------------------------------
def inference(model, X):
    logits = model(X)
    probabilities = torch.sigmoid(logits)
    return logits, probabilities


def predict_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        test_preds = []
        for i, X in enumerate(dataloader):
            X = X[0].cuda().long()  # X[0]: text sequences, X[1]: lengths
            logits, probabilities = inference(model, X)
            test_preds.append(probabilities.cpu().numpy())
    test_preds = np.concatenate(test_preds)
    return test_preds


# RNN prediction1 ------------------------------------------------------------------------
test_loader = prepare_data_loader(X_test, test_lengths, shuffle=False)

pred_tests = []
weights = sorted(glob.glob('../input/toxicfittedmodels3/rnn_weight_best_fold_*.pth'))
for fold, path in enumerate(weights):
    model = EmbLSTM(embedding_matrix, max_features).cuda()
    state = torch.load(path)
    state['state_dict']['embedding.weight'] = torch.tensor(embedding_matrix)
    model.load_state_dict(state['state_dict'])

    pred_test = predict_model(model, test_loader)
    pred_tests.append(pred_test)

    del model;
    gc.collect();
    torch.cuda.empty_cache()

pred_tests_rnn = np.mean(pred_tests, axis=0)
# RNN prediction2 ------------------------------------------------------------------------
pred_tests = []
weights = sorted(glob.glob('../input/toxicfittedmodels3/rnn_trans_weight_best_fold_*.pth'))
for fold, path in enumerate(weights):
    model = EmbLSTM(embedding_matrix, max_features).cuda()
    state = torch.load(path)
    state['state_dict']['embedding.weight'] = torch.tensor(embedding_matrix)
    model.load_state_dict(state['state_dict'])

    pred_test = predict_model(model, test_loader)
    pred_tests.append(pred_test)

    del model;
    gc.collect();
    torch.cuda.empty_cache()

pred_tests_rnn_trans = np.mean(pred_tests, axis=0)
# RNN prediction3 ------------------------------------------------------------------------
pred_tests = []
weights = sorted(glob.glob('../input/toxicfittedmodels3/jigsaw_old_weight_best_fold_*.pth'))
for fold, path in enumerate(weights):
    model = EmbLSTM(embedding_matrix, max_features).cuda()
    state = torch.load(path)
    state['state_dict']['embedding.weight'] = torch.tensor(embedding_matrix)
    model.load_state_dict(state['state_dict'])

    pred_test = predict_model(model, test_loader)
    pred_tests.append(pred_test)

    del model;
    gc.collect();
    torch.cuda.empty_cache()

pred_tests_rnn_old = np.mean(pred_tests, axis=0)

######################################## BERT Part #######################################
# BERT -----------------------------------------------------------------------------------
# constants
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
bert_config = BertConfig('../input/toxicfittedmodels/bert_config.json')

# settings
seed = 0
device = "cuda:0"
n_labels = len(AUX_COLUMNS) + 1
# n_labels = 1
max_len = 220
head_len = 80
batch_size = 32 * 16
seed_torch(seed)
# ----------------------------------------------------------------------------------------
test['comment_text'] = test['comment_text'].astype(str)
test = test.fillna(0)
# ----------------------------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=True)
X_text = convert_lines_head_tail(test['comment_text'].fillna("DUMMY_VALUE").values, max_len, head_len, tokenizer)
# ----------------------------------------------------------------------------------------
index = np.argsort(test_lengths)
inverse_index = np.argsort(np.arange(len(test))[index])


# prediction -----------------------------------------------------------------------------
def inference(models, test_loader, device, n_labels):
    test_pred = []
    with torch.no_grad():
        for features in test_loader:
            features = trim_tensors(features[0])
            features = features.to(device)
            for i, model in enumerate(models):
                if i == 0:
                    logits = torch.sigmoid(model(features, attention_mask=features > 0, labels=None))
                else:
                    logits += torch.sigmoid(model(features, attention_mask=features > 0, labels=None))
            test_pred.append(logits / len(models))

        test_pred = torch.cat(test_pred).float().cpu().numpy()

    return test_pred


# normal ---------------------------------------------------------------------------------
X_text = X_text[index]
# ----------------------------------------------------------------------------------------
model_paths = sorted(glob.glob('../input/toxicfittedmodels3/exp12_bert_epoch1_fold*.pth'))
# ----------------------------------------------------------------------------------------
models = []
for model_path in model_paths:
    model = BertForSequenceClassification(bert_config, num_labels=n_labels)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    models.append(model)

test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_text, dtype=torch.long))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

pred_tests_bert = inference(models, test_loader, device, n_labels)
del models
gc.collect()
torch.cuda.empty_cache()
# ----------------------------------------------------------------------------------------
pred_tests_bert = pred_tests_bert[inverse_index]
# cased ----------------------------------------------------------------------------------
# constants
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
BERT_MODEL_PATH = '../input/bert-pretrained-models/cased_l-12_h-768_a-12/cased_L-12_H-768_A-12/'
bert_config = BertConfig('../input/jigsaw-exp15-weights/bert_config.json')

# settings
seed = 0
device = "cuda:0"
n_labels = len(AUX_COLUMNS) + 1
# n_labels = 1
max_len = 220
head_len = 80
batch_size = 32 * 16
seed_torch(seed)
# ----------------------------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=False)
X_text = convert_lines_head_tail(test['comment_text'].fillna("DUMMY_VALUE").values, max_len, head_len, tokenizer)
# ----------------------------------------------------------------------------------------
index = np.argsort(test_lengths)
inverse_index = np.argsort(np.arange(len(test))[index])
# ----------------------------------------------------------------------------------------
X_text = X_text[index]
# ----------------------------------------------------------------------------------------
model_paths = sorted(glob.glob('../input/jigsaw-exp15-weights/exp14_bert_base_cased_epoch1_fold*.pth'))
# ----------------------------------------------------------------------------------------
models = []
for model_path in model_paths:
    model = BertForSequenceClassification(bert_config, num_labels=n_labels)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    models.append(model)

test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_text, dtype=torch.long))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

pred_tests_bert_cased = inference(models, test_loader, device, n_labels)
del models
gc.collect()
torch.cuda.empty_cache()
# ----------------------------------------------------------------------------------------
pred_tests_bert_cased = pred_tests_bert_cased[inverse_index]
# multi cased ----------------------------------------------------------------------------
# constants
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
BERT_MODEL_PATH = '../input/bert-pretrained-models/multi_cased_l-12_h-768_a-12/multi_cased_L-12_H-768_A-12/'
bert_config = BertConfig(BERT_MODEL_PATH + 'bert_config.json')

# settings
seed = 0
device = "cuda:0"
n_labels = len(AUX_COLUMNS) + 1
# n_labels = 1
max_len = 220
head_len = 80
batch_size = 32 * 16
seed_torch(seed)
# ----------------------------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=False)
X_text = convert_lines_head_tail(test['comment_text'].fillna("DUMMY_VALUE").values, max_len, head_len, tokenizer)
# ----------------------------------------------------------------------------------------
index = np.argsort(test_lengths)
inverse_index = np.argsort(np.arange(len(test))[index])
# ----------------------------------------------------------------------------------------
X_text = X_text[index]
# ----------------------------------------------------------------------------------------
model_paths = sorted(glob.glob('../input/jigsaw-exp15-multi-weights/exp15_bert_base_multi_cased_epoch1_fold*.pth'))
# ----------------------------------------------------------------------------------------
models = []
for model_path in model_paths:
    model = BertForSequenceClassification(bert_config, num_labels=n_labels)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    models.append(model)

test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_text, dtype=torch.long))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

pred_tests_bert_multi_cased = inference(models, test_loader, device, n_labels)
del models
gc.collect()
torch.cuda.empty_cache()
# ----------------------------------------------------------------------------------------
pred_tests_bert_multi_cased = pred_tests_bert_multi_cased[inverse_index]
# submission -----------------------------------------------------------------------------
del embedding_matrix;
gc.collect()


# ----------------------------------------------------------------------------------------
def load_oof_predictions(paths):
    pred = np.zeros((1804874,))
    folds = pd.read_csv('../input/toxicstratifiedkfold/fold01.csv')
    for i, path in zip(sorted(folds['fold_id'].unique()), paths):
        mask = folds['fold_id'] == i
        oof = np.load(path)
        if len(oof.shape) > 1:
            pred[mask] = oof[:, 0]
        else:
            pred[mask] = oof.flatten()
    return pred


# ----------------------------------------------------------------------------------------
# Tekito Na Omomi
w = [0.39193463, 0.14464678, 0.03892303, 0.06871146, 0.317778]
y_pred = pred_tests_bert * w[0] + pred_tests_rnn * w[1] + pred_tests_rnn_trans * w[2] + \
         pred_tests_rnn_old * w[3] + pred_tests_bert_cased * w[4] + pred_tests_bert_multi_cased * (1 - np.sum(w))


# ----------------------------------------------------------------------------------------
def submission(ids, y_pred):
    sub = pd.DataFrame(None, columns=['id', 'prediction'])
    sub['id'] = ids
    sub['prediction'] = y_pred[:, 0]
    sub.to_csv('submission.csv', index=False)
    return sub


# ----------------------------------------------------------------------------------------
sub = submission(test['id'], y_pred)
