import re
from typing import List, Optional

# تلاش برای وارد کردن کتابخانه هضم برای پردازش متن فارسی
try:
    import hazm

    hazm_available = True
except ImportError:
    hazm_available = False


def extract_keywords(text: str) -> List[str]:
    """
    استخراج کلمات کلیدی از متن توییت

    :param text: متن توییت
    :return: لیست کلمات کلیدی استخراج شده
    """
    keywords = []

    # استخراج هشتگ‌ها
    hashtags = extract_hashtags(text)
    if hashtags:
        keywords.extend(hashtags)

    # استخراج کلمات کلیدی با استفاده از هضم (اگر در دسترس باشد)
    if hazm_available:
        keywords.extend(extract_keywords_with_hazm(text))

    # حذف تکرارها و استانداردسازی
    unique_keywords = list(set([k.lower() for k in keywords]))

    return unique_keywords


def extract_hashtags(text: str) -> List[str]:
    """
    استخراج هشتگ‌ها از متن

    :param text: متن توییت
    :return: لیست هشتگ‌ها
    """
    if not text:
        return []

    # الگوی یافتن هشتگ
    hashtag_pattern = r'#(\w+)'

    # جستجو برای هشتگ‌ها
    hashtags = re.findall(hashtag_pattern, text)

    return hashtags


def extract_keywords_with_hazm(text: str) -> List[str]:
    """
    استخراج کلمات کلیدی با استفاده از کتابخانه هضم

    :param text: متن توییت
    :return: لیست کلمات کلیدی
    """
    if not hazm_available:
        return []

    try:
        # نرمال‌سازی متن
        normalizer = hazm.Normalizer()
        normalized_text = normalizer.normalize(text)

        # توکن‌سازی
        tokens = hazm.word_tokenize(normalized_text)

        # برچسب‌گذاری اجزای کلام
        tagger = hazm.POSTagger(model='resources/postagger.model')
        tagged_tokens = tagger.tag(tokens)

        # استخراج اسامی و صفات به عنوان کلمات کلیدی احتمالی
        keywords = []
        for word, tag in tagged_tokens:
            if tag.startswith('N') or tag.startswith('ADJ'):  # اسم یا صفت
                if len(word) > 2:  # حذف کلمات کوتاه
                    keywords.append(word)

        return keywords
    except Exception:
        # در صورت بروز خطا، لیست خالی برگردانید
        return []


def clean_text(text: str) -> str:
    """
    پاکسازی و استانداردسازی متن

    :param text: متن ورودی
    :return: متن پاکسازی شده
    """
    if not text:
        return ""

    # حذف لینک‌ها
    text = re.sub(r'https?://\S+', '', text)

    # حذف منشن‌ها
    text = re.sub(r'@\w+', '', text)

    # حذف کاراکترهای خاص و فاصله‌های اضافی
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)  # حفظ کاراکترهای فارسی
    text = re.sub(r'\s+', ' ', text).strip()

    # اگر هضم در دسترس باشد، از نرمال‌سازی آن استفاده کنید
    if hazm_available:
        normalizer = hazm.Normalizer()
        text = normalizer.normalize(text)

    return text


def detect_sentiment(text: str) -> Optional[float]:
    """
    تشخیص احساسات متن

    :param text: متن ورودی
    :return: امتیاز احساسات (-1 تا 1) یا None در صورت خطا
    """
    # این فقط یک نمونه ساده است و باید با یک مدل واقعی جایگزین شود

    # لیستی از کلمات منفی فارسی
    negative_words = [
        'بد', 'افتضاح', 'مزخرف', 'ضعیف', 'زشت', 'وحشتناک', 'متنفر',
        'اشتباه', 'خراب', 'ناراحت', 'عصبانی', 'غم', 'فاجعه', 'شکست',
        'نامناسب', 'ناقص', 'غیرممکن', 'نا', 'بدون', 'ضد'
    ]

    # لیستی از کلمات مثبت فارسی
    positive_words = [
        'خوب', 'عالی', 'فوق‌العاده', 'محشر', 'زیبا', 'قشنگ', 'دوست',
        'موفق', 'مناسب', 'کامل', 'لذت', 'شاد', 'خوشحال', 'پیروز',
        'عشق', 'امید', 'بهتر', 'بهترین'
    ]

    # پاکسازی متن
    clean = clean_text(text)

    # شمارش کلمات مثبت و منفی
    positive_count = sum(1 for word in positive_words if word in clean.split())
    negative_count = sum(1 for word in negative_words if word in clean.split())

    # محاسبه امتیاز احساسات
    total_count = positive_count + negative_count

    if total_count == 0:
        return 0  # خنثی

    sentiment_score = (positive_count - negative_count) / total_count

    return sentiment_score