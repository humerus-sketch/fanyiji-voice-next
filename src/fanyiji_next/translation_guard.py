"""Conservative, offline checks for conspicuous translation inconsistencies.

These rules are a fallback gate, not a semantic proofreader. They never repair a
candidate translation or infer a missing unit, referent, or negative particle.
"""

from collections import Counter
from decimal import Decimal, InvalidOperation
import re
import unicodedata


_DIGITS = {char: str(value) for value, chars in enumerate(
    ("零〇", "一壹", "二两兩贰貳", "三叁參", "四肆", "五伍", "六陆陸", "七柒", "八捌", "九玖")
) for char in chars}
_SMALL_UNITS = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
_LARGE_UNITS = {"万": 10_000, "萬": 10_000, "亿": 100_000_000,
                "億": 100_000_000, "兆": 1_000_000_000_000}
_HAN_NUMBER = "".join(_DIGITS) + "".join(_SMALL_UNITS) + "".join(_LARGE_UNITS)
_HAN_RUN = re.compile(rf"[{_HAN_NUMBER}]+(?:[点點][{_HAN_NUMBER}]+)?")
_HAN_CLOCK = re.compile(rf"([{_HAN_NUMBER}]+)([点點])([{_HAN_NUMBER}]+)分(?![钟鐘間])")
_JA_DECIMAL_DIGITS = "零〇一二三四五六七八九0-9０-９"
_JA_MIDDLE_DOT_DECIMAL = re.compile(
    rf"(?<![{_HAN_NUMBER}0-9０-９・･A-Za-z_])"
    rf"([{_HAN_NUMBER}]+|[0-9０-９]+)[・･]([{_JA_DECIMAL_DIGITS}]+)"
    r"(キログラム|ミリグラム|グラム|キロメートル|センチメートル|ミリメートル|メートル|"
    r"ミリリットル|リットル|パーセント)(?![ァ-ヿA-Za-z0-9_])"
)
# Single digits need an explicit numeric context; otherwise 一緒/一般/一旦
# and countless ordinary words would be changed by a numeral-only regex.
_QUANTITY_SUFFIX = re.compile(
    r"(?:年|月|日|号|號|时|時|点|點|秒|分钟|分鐘|分間|分|小時|小时|"
    r"元|圆|圓|円|块|塊|美元|美金|日元|日圓|人民币|人民幣|"
    r"个|個|件|张|張|枚|台|次|回|份|支|瓶|箱|公斤|千克|克|公里|米|"
    r"公升|毫升|升|度|歳|岁|歲|人|名|本|冊|册|つ目|番目|%|％)"
)
_ARABIC_RUN = re.compile(r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?")
_CLAUSE_BREAK = re.compile(r"[，,。.!！;；\n]+")
_JA_CLAUSE_BREAK = re.compile(r"[。.!！;；\n]+")
_PER_UNIT_FORMS = {
    "person": (r"每(?:个|個)?人|各人|每位|每名", r"人|名"),
    "household": (r"每(?:个|個)?家庭|每户|每戶|各户|各戶", r"家庭|世帯|戸"),
    "group": (r"每(?:个|個)?[组組]|各[组組]", r"组|組"),
    "item": (r"每件|各件", r"件"),
    "machine": (r"每台|各台", r"台"),
    "sheet": (r"每[张張]|各[张張]", r"枚|張"),
    "ticket": (r"每票|各票", r"票"),
    "box": (r"每箱|各箱", r"箱"),
    "bottle": (r"每瓶|各瓶", r"瓶|ボトル"),
    "book": (r"每[册冊]|各[册冊]", r"冊|册"),
}
# 最初から means from the beginning, not the first numbered attempt. Keep
# explicit first-attempt words distinct from a real amount or repeat count.
_FIRST_POSITION = re.compile(r"最初(?!から)|首先|起初|首次|初回|首回")
_JA_FRAGMENT_SEPARATORS = re.compile(r"[\s、，,。.!！?？…・]+")
_JA_RHYTHMIC_TAPS = re.compile(
    r"(?P<beat>[たてだで])(?:っ?(?P=beat)){2,}っ?"
)
_JA_REPEATED_HISS = re.compile(r"(?:し[ー〜~]+っ?){2,}")


def source_fragment_issue(source: str, direction: str) -> str | None:
    """Identify only narrow phonetic repetitions with unresolved semantics.

    This is not a kana-only, short-sentence, name or repetition detector. Words
    such as ナイス, やった, はいはい, ねーねー and ヨーヨー must still reach MT.
    A matched fragment is retained verbatim, never deduplicated or guessed into
    an emotion, affectionate address or action based on the prior two lines.
    """
    if direction != "ja_to_zh":
        return None
    normalized = unicodedata.normalize("NFKC", source)
    normalized = "".join(chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char
                         for char in normalized)
    compact = _JA_FRAGMENT_SEPARATORS.sub("", normalized)
    if ((_JA_RHYTHMIC_TAPS.fullmatch(compact) and compact.count("っ") >= 2)
            or _JA_REPEATED_HISS.fullmatch(compact)):
        return "unresolved_phonetic_fragment"
    return None


def _quality_text(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", text)
                   if not char.isspace())


def _language_repeats(text: str):
    """Yield primitive short cycles without treating numerals as words."""
    for width in range(1, 9):
        start = 0
        while start + width * 6 <= len(text):
            unit = text[start:start + width]
            end = start + width
            while text.startswith(unit, end):
                end += width
            count = (end - start) // width
            # Numeric identifiers can legitimately consist of the same digit,
            # including a spelled-out Han digit. Number checks remain separate.
            language = any(unicodedata.category(char).startswith("L")
                           and char not in _HAN_NUMBER for char in unit)
            primitive = not any(width % size == 0 and unit == unit[:size] * (width // size)
                                for size in range(1, width))
            if count >= 6 and language and primitive:
                yield count, end - start
            start = end if count >= 2 else start + 1


def _dominant_language_repeat(text: str) -> bool:
    return any(span * 5 >= len(text) * 4 for _, span in _language_repeats(text))


def _supports_embedded_repeat(source: str, target: str) -> bool:
    source_repeats = [(count, span) for count, span in _language_repeats(source) if span >= 12]
    return any(source_count >= target_count and target_span <= source_span * 2
               for target_count, target_span in _language_repeats(target)
               if target_span * 5 >= len(target) * 4
               for source_count, source_span in source_repeats)


def output_quality_issues(source: str, target: str) -> tuple[str, ...]:
    """Reject conspicuous generation degeneration without rewriting a result.

    This deliberately permits normal emphasis, laughter, Japanese text and long
    digit strings. Explicit, dominant repetition in the source also makes a
    repetitive translation inconclusive. Passing this check proves no meaning.
    """
    output = _quality_text(target)
    original = _quality_text(source)
    if output == original:
        return ()
    raw_output = "".join(char for char in target if not char.isspace())
    unknown_count = raw_output.count("\u2047")
    source_unknown_count = source.count("\u2047")
    if len(output) >= 12 and all(
        unicodedata.category(char).startswith("P") or char == "\ufffd" for char in output
    ):
        source_supports_symbols = unknown_count >= 3 and unknown_count <= source_unknown_count
        if not source_supports_symbols:
            return ("output_invalid_characters" if "\ufffd" in output else "output_punctuation",)
    # SentencePiece's unknown surface can be U+2047. NFKC expands it to "??",
    # hiding that evidence; one stray decoded word must not legitimize the run.
    if (unknown_count >= 3 and unknown_count * 2 >= len(raw_output)
            and unknown_count > source_unknown_count):
        return ("output_invalid_characters",)
    if len(output) < 12:
        return ()
    question_count = output.count("?")
    if (question_count >= 12 and question_count * 5 >= len(output) * 4
            and question_count > original.count("?") * 2):
        return ("output_punctuation",)
    if (output.count("\ufffd") >= 12
            and output.count("\ufffd") * 5 >= len(output) * 4):
        return ("output_invalid_characters",)
    if (_dominant_language_repeat(output) and not _dominant_language_repeat(original)
            and not _supports_embedded_repeat(original, output)):
        return ("output_repetition",)
    return ()


_JA_TIME_DEICTIC = re.compile(r"こっち|そっち|あっち|こちら|そちら|あちら")
_JA_CURRENT_TIME = re.compile(
    rf"[0-9{_HAN_NUMBER}]+(?:年|月|日|時|分|秒)|[0-9]{{1,2}}:[0-9]{{2}}|"
    r"今日|明日|昨日|一昨日|明後日|今朝|今夜|今晩|昨夜|昨晩|[月火水木金土日]曜日"
)
_TIME_REFERENCE_FAMILIES = tuple((re.compile(chinese), re.compile(japanese))
    for chinese, japanese in (
        (r"上午|早上|早晨|清晨|早间|早間", r"午前|朝|明け方|おはよう"),
        (r"中午|午间|午間|正午", r"昼|正午"),
        (r"白天|日间|日間", r"昼|日中|午前|朝|午後"),
        (r"下午(?!茶)|午后|午後", r"午後|昼下がり"),
        (r"傍晚|黄昏|黃昏", r"夕方|夕暮|夕刻|夕焼|黄昏"),
        # Chinese 晚上 spans evening and night; do not reject that natural
        # generalization when the source explicitly supplies either period.
        (r"晚上|晚间|晚間", r"夕方|夕暮|夕刻|夕焼|黄昏|夜|晩"),
        (r"夜里|夜裡|夜间|夜間|夜晚|深夜|半夜|午夜|凌晨", r"夜|晩|未明"),
    ))
_CHINESE_TITLE = re.compile(r"《[^》]*》|〈[^〉]*〉")


def context_grounding_issues(source: str, target: str, prior) -> tuple[str, ...]:
    """Check only invented Chinese dayparts for Japanese deictic utterances.

    The JA-to-ZH caller supplies the actual source context used for this
    candidate, or no context for an independent fallback. This checks that a
    period has some basis; it never decides who/what a deictic refers to, which
    grounded period it selects, or whether the rest of a translation is right.
    Explicit current dates, clocks or dayparts deliberately remain outside it.
    """
    current = unicodedata.normalize("NFKC", source)
    if (not _JA_TIME_DEICTIC.search(current) or _JA_CURRENT_TIME.search(current)
            or any(japanese.search(current) for _, japanese in _TIME_REFERENCE_FAMILIES)):
        return ()
    candidate = _CHINESE_TITLE.sub("", unicodedata.normalize("NFKC", target))
    grounding = "\n".join(unicodedata.normalize("NFKC", text)
                          for text in (*tuple(prior)[-2:], source))
    if any(chinese.search(candidate) and not japanese.search(grounding)
           for chinese, japanese in _TIME_REFERENCE_FAMILIES):
        return ("ungrounded_time_reference",)
    return ()


def _parse_han_integer(value: str) -> str | None:
    if not value:
        return None
    if all(char in _DIGITS for char in value):
        return "".join(_DIGITS[char] for char in value)
    large = [(unit, amount) for unit, amount in _LARGE_UNITS.items() if unit in value]
    if large:
        unit, amount = max(large, key=lambda item: item[1])
        if value.count(unit) != 1:
            return None
        left, right = value.split(unit)
        left_number = _parse_han_integer(left) if left else "1"
        right_number = _parse_han_integer(right) if right else "0"
        # 一万二 / 一百二 can mean an omitted lower unit. Do not guess.
        if right and len(right) == 1 and right in _DIGITS and right not in "零〇":
            return None
        if left_number is None or right_number is None or int(right_number) >= amount:
            return None
        return str(int(left_number) * amount + int(right_number))
    total = 0
    pending = None
    last_unit = 10_000
    zero_after_unit = False
    for char in value:
        if char in _DIGITS:
            digit = int(_DIGITS[char])
            if pending is not None and pending != 0:
                return None
            if digit == 0:
                zero_after_unit = True
            pending = digit
            continue
        unit = _SMALL_UNITS.get(char)
        if unit is None or unit >= last_unit or pending == 0:
            return None
        total += (1 if pending is None else pending) * unit
        pending = None
        last_unit = unit
        zero_after_unit = False
    if pending is not None:
        if pending and last_unit > 10 and not zero_after_unit:
            return None
        total += pending
    return str(total)


def _is_numeric_context(text: str, start: int, end: int, token: str) -> bool:
    before, after = text[:start], text[end:]
    if before.endswith("百分之"):
        return True
    if token == "百" and after.startswith(("分之", "分比", "分率", "分点", "分點")):
        return False
    if token in _DIGITS:
        literal_end = not after or bool(re.match(r"[\s，,。.!！?？;；:：”」』\"']|(?:也|都|要|仍|和|与|與|及|或|不|先|请|請)", after))
        if literal_end and re.search(rf"[{_HAN_NUMBER}0-9]+(?:个|個|つの|つ|枚の)\s*$", before):
            return True
        if token in {"零", "〇"}:
            quoted = bool(before and before[-1] in "“「『\"'" and after and after[0] in "”」』\"'")
            if quoted or (literal_end and re.search(
                r"(?:数字|數字|字符|字面|(?:编号|編號|番号)(?:为|為|是|は)?|"
                r"(?:数量|數量|数值|數值|计数|計數|金额|金額|余额|餘額)(?:为|為|是|等于|等於|：|:))\s*$", before
            )):
                return True
            # 零件 is a noun, even in 三个零件. Count a literal zero only
            # with a numeric reference boundary; never split ordinary words.
            if after.startswith("件"):
                return False
            return bool(_QUANTITY_SUFFIX.match(after)) or (not before.strip() and not after.strip())
    if len(token) >= 2 and all(char in _DIGITS for char in token) and (
        re.search(r"(?:编号|編號|编码|編碼|号码|號碼|单号|單號|写成|寫成|读作|讀作|番号)(?:是|为|為|は)?\s*$", before) or
        re.match(r"(?:と|って|という)?(?:書|写|寫|记|記)", after)
    ):
        return True
    if token == "一" and (
        (re.search(r"(?:もう|また|再び)\s*$", before) and re.match(r"(?:度|回)(?!だけ|限り)", after)) or
        (after.startswith("次") and
         re.search(rf"(?:再|重新)[^{_HAN_NUMBER}0-9，,。.!！?？;；\n]{{0,12}}$", before) and
         not re.search(r"(?:只|仅|僅)[^，,。.!！?？;；\n]{0,16}$", before))
    ):
        # もう一度 / 再一次 can mean simply "again", with no explicit count
        # in an equally valid translation. Keep exact 一度だけ counts protected.
        return False
    if token == "千万" and re.match(r"(?:不要|别|別|不|注意|小心|记住|記住|务必|務必)", after):
        return False
    if token == "十" and after.startswith("分"):
        # 十分注意/十分重要 are intensifiers, not a ten-minute quantity.
        tail = after[1:]
        if tail and not re.match(r"(?:钟|鐘|間|后|後|前|以内|以內|左右|整|[\s，,。.!！?？;；:：])", tail):
            return False
    if token == "一" and after.startswith(("点", "點")):
        if not (re.search(r"(?:上午|下午|凌晨|早上|晚上|中午|深夜|傍晚)\s*$", before) or
                re.match(r"[点點](?:钟|鐘|整|半|前|后|後|出发|出發|到)", after)):
            return False
    if _QUANTITY_SUFFIX.match(after) or before.endswith(("第", "¥", "￥", "$", "百分之")):
        return True
    if "点" in token or "點" in token:
        return True
    if len(token) > 1 and any(char in _SMALL_UNITS or char in _LARGE_UNITS for char in token):
        # 万一 is a connective; the parser also rejects the ambiguous trailing 一.
        return _parse_han_integer(token) is not None
    if len(token) >= 2 and any(char in "零〇" for char in token):
        return True
    return not before.strip() and not after.strip()


def prepare_source(text: str) -> str:
    """Normalize only unambiguous Han numeric spans, preserving all other text."""
    def replace_middle_dot(match: re.Match[str]) -> str:
        before = text[:match.start()]
        # A measurement in a quoted name/code is not necessarily a quantity.
        if any(before.rfind(opening) > before.rfind(closing) for opening, closing in (
            ("「", "」"), ("『", "』"), ("“", "”")
        )) or before.count('"') % 2:
            return match.group()
        integer = unicodedata.normalize("NFKC", match[1])
        if not integer.isdecimal():
            integer = _parse_han_integer(integer)
        if integer is None:
            return match.group()
        fraction = "".join(_DIGITS.get(char, char) for char in unicodedata.normalize("NFKC", match[2]))
        return integer + "." + fraction + match[3]

    # A lone Japanese middle dot immediately before an explicit measurement
    # unit can denote a decimal. Never interpret dot-separated lists as one.
    text = _JA_MIDDLE_DOT_DECIMAL.sub(replace_middle_dot, text)

    def replace_clock(match: re.Match[str]) -> str:
        hour = _parse_han_integer(match[1])
        minute = _parse_han_integer(match[3])
        if hour is None or minute is None or not (0 <= int(hour) <= 23 and 0 <= int(minute) <= 59):
            return match.group()
        return hour + match[2] + minute + "分"

    # 三点十五分 is an explicit clock time, unlike 三点五元 / 三点五分钟.
    text = _HAN_CLOCK.sub(replace_clock, text)

    def replace(match: re.Match[str]) -> str:
        token = match.group()
        before, after = text[:match.start()], text[match.end():]
        if (re.search(rf"[{_HAN_NUMBER}0-9０-９][・･]$", before) or
                re.match(rf"[・･][{_HAN_NUMBER}0-9０-９]", after)):
            # Keep unresolved middle-dot runs intact. Otherwise an ambiguous
            # 二・十五 could become 二・15, then 2.15 on the next call.
            return token
        # Replacing only 万 inside 1万元 would concatenate 1 and 10000.
        if ((match.start() and text[match.start() - 1].isdigit()) or
                (match.end() < len(text) and text[match.end()].isdigit())):
            return token
        if not _is_numeric_context(text, match.start(), match.end(), token):
            return token
        parts = re.split(r"[点點]", token)
        integer = _parse_han_integer(parts[0])
        if integer is None:
            return token
        if len(parts) == 2:
            if not all(char in _DIGITS for char in parts[1]):
                return token
            return integer + "." + "".join(_DIGITS[char] for char in parts[1])
        return integer

    return _HAN_RUN.sub(replace, text)


def _has_ambiguous_numeric_span(text: str) -> bool:
    for fraction in re.finditer(rf"[{_HAN_NUMBER}]+分(?:之|の)[{_HAN_NUMBER}0-9]+", text):
        if not fraction.group().startswith("百分之"):
            return True
    text = prepare_source(text)
    if re.search(rf"[{_HAN_NUMBER}0-9０-９][・･][{_HAN_NUMBER}0-9０-９]", text):
        # Remaining middle dots may be lists, names or malformed decimals;
        # avoid proving a numeric mismatch from one guessed interpretation.
        return True
    for match in _HAN_RUN.finditer(text):
        token = match.group()
        before, after = text[:match.start()], text[match.end():]
        mixed_left = bool(before and before[-1].isdigit())
        mixed_right = bool(after and after[0].isdigit())
        if mixed_left or mixed_right:
            # Simple 1万元 / 3.5千円 are interpreted below. More complicated
            # mixed forms such as 1万2千 are left to the translator.
            if mixed_left and not mixed_right and token in (_SMALL_UNITS | _LARGE_UNITS):
                continue
            return True
        if _is_numeric_context(text, match.start(), match.end(), token):
            parts = re.split(r"[点點]", token)
            if (_parse_han_integer(parts[0]) is None or
                    (len(parts) == 2 and not all(char in _DIGITS for char in parts[1]))):
                return True
    return False


def _implicit_per_units(text: str) -> frozenset[str]:
    return frozenset(kind for kind, (implicit, _) in _PER_UNIT_FORMS.items()
                     if re.search(implicit, text))


def _numeric_values(text: str, *, per_units: frozenset[str] = frozenset(), positions: bool = False) -> Counter[str]:
    text = unicodedata.normalize("NFKC", prepare_source(text))
    # Half past an explicit clock hour is canonically hour + 30 minutes. This
    # is an internal comparison view; prepare_source never inserts a new unit.
    text = re.sub(r"([0-9]{1,2})([時时点點])半(?![会會])",
                  lambda match: match[1] + match[2] + "30分" if int(match[1]) <= 23 else match.group(), text)
    values: Counter[str] = Counter()
    if positions:
        values["position:1"] += len(_FIRST_POSITION.findall(text))
    date_spans = [match.span() for match in re.finditer(r"(?<![0-9])[0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}(?![0-9])", text)]
    for match in _ARABIC_RUN.finditer(text):
        raw = match.group().replace(",", "")
        before, after = text[:match.start()], text[match.end():]
        if raw == "1" and any(
            re.match(rf"(?:{_PER_UNIT_FORMS[kind][1]})(?:あたり|当たり|当り|につき|に(?:入れる数|入る数|入っている数|含まれる数|詰める数|[0-9]))", after)
            for kind in per_units
        ):
            # Every person / each group permits an explicit one-unit
            # denominator in the other language, not an extra quota or amount.
            continue
        if positions and raw.isdigit() and (
            (before.endswith("第") and re.match(r"(?:次|回)", after)) or re.match(r"(?:回|度)目", after)
        ):
            # Sequence position has its own key, so 最初/第一次 cannot
            # consume a real one-yen amount or hide a changed two-times count.
            values["position:" + str(int(raw))] += 1
            continue
        sign = ""
        if before.endswith(("负", "負", "マイナス")):
            sign = "-"
        elif before.endswith("-") and (len(before) < 2 or not before[-2].isdigit()):
            sign = "-"
        multiplier_match = None if after.startswith(("百分比", "百分率", "百分点", "百分點")) else re.match(
            rf"([{''.join(_SMALL_UNITS)}{''.join(_LARGE_UNITS)}])(?![{_HAN_NUMBER}0-9])", after)
        multiplier = (_SMALL_UNITS | _LARGE_UNITS)[multiplier_match[1]] if multiplier_match else 1
        quantity = bool(_QUANTITY_SUFFIX.match(after)) or before.endswith(("¥", "￥", "$")) or multiplier != 1
        date_piece = any(start <= match.start() and match.end() <= end for start, end in date_spans)
        # An unqualified 001 is an identifier, unlike 01月 or 001円.
        if len(raw) > 1 and raw.startswith("0") and "." not in raw and not quantity and not date_piece:
            values["literal:" + sign + raw] += 1
            continue
        try:
            number = Decimal(sign + raw) * multiplier
            values[str(number.normalize()) if number else "0"] += 1
        except InvalidOperation:
            continue
    return values


_ZH_PROHIBITION = re.compile(r"请勿|請勿|切勿|严禁|嚴禁|禁止|不得|不准|不许|不許|不要|(?<!特)别|(?<!特)別")
_ZH_TARGET_PROHIBITION = re.compile(r"请勿|請勿|切勿|严禁|嚴禁|禁止|不得|不准|不许|不許|不能|不可|不要|(?<!特)别|(?<!特)別")
_JA_PROHIBITION = re.compile(
    r"禁止|禁じ|ては(?:いけま?せ?ん|いけない|ならない|なりません)|"
    r"ないで(?:ください|下さい|くれ)|ないように(?:してください|して下さい)|"
    r"(?:お控え|控えて)ください|"
    # The final うな in だろうな/でしょうな expresses conjecture, not a
    # dictionary-form verb followed by the prohibitive particle な.
    r"ないで\s*$|(?<!だろ)(?<!でしょ)[うくぐすつぬぶむる]な\s*$"
)
_JA_STOP_REQUEST = re.compile(r"(?:やめて|止めて|おやめ)(?:ください|下さい)")


def _has_conjunctive_negative_request(clause: str) -> bool:
    """Recognize a negative verb joined to a later explicit request, not all ず."""
    for match in re.finditer(r"[ぁ-ゖ]ず(?:に)?", clause):
        before, after = clause[:match.end()], clause[match.end():]
        if re.search(r"(?:必ず|まず|はず|絶えず|思わず|迷わず|惜しまず|欠かさず|限らず|問わず|相変わらず|残らず|知らず知らず)(?:に)?$", before):
            continue
        if after.startswith(("つ", "っと")) or (before.endswith("いず") and after.startswith("れ")):
            continue
        if re.match(r"(?:済|すん|すむ|いた|いました|いる|いたの|だった|でした)", after):
            continue
        request = re.search(r"(?:て|で)(?:ください|下さい)", after)
        if request and not re.search(r"(?:済んだ|済みました|済む|すんだ|すみました|だった|でした)", after[:request.end()]):
            return True
    return False


def _is_conditional_ban(marker: str, after: str) -> bool:
    # 不要不商量就决定 prohibits the later action without a prerequisite.
    # 不得不 remains an obligation, and 不要不付款 alone remains ambiguous.
    return marker in {"不要", "请勿", "請勿", "不准", "不许", "不許", "别", "別"} and bool(
        re.match(r"\s*(?:不|未|没|沒)[^，,。.!！?？;；\n]{1,24}?(?:就|便)[^，,。.!！?？;；\n]+", after)
    )


def _has_prohibition(text: str, language: str, *, candidate: bool = False) -> bool:
    splitter = _JA_CLAUSE_BREAK if language == "ja" else _CLAUSE_BREAK
    for clause in splitter.split(text):
        if not clause.strip():
            continue
        if language == "zh":
            pattern = _ZH_TARGET_PROHIBITION if candidate else _ZH_PROHIBITION
            for match in pattern.finditer(clause):
                before, after = clause[:match.start()], clause[match.end():]
                if match.group() == "不要" and before.endswith("要"):
                    # 要不要 remains a question even when a following comma
                    # separates this fragment from the final question mark.
                    continue
                if re.search(r"能不能(?:先|暂时|暫時)?\s*$", before):
                    continue
                if re.search(
                    r"的?(?:指令|命令|禁令|要求)(?:已经|已經|现已|現已|已)(?:被)?(?:取消|撤销|撤銷|解除|作废|作廢|失效)", after
                ):
                    # This reports an explicitly revoked instruction; a later
                    # current prohibition is still inspected independently.
                    continue
                # 不得不 / 不许不 / 请勿不 negate a negative action. They
                # cannot establish a prohibition on that action in either role.
                if re.match(r"\s*(?:不|未|没|沒|勿|莫|别|別)", after):
                    # A conditional ban is a source directive, but as a target
                    # it needs a matching conditional structure below. It must
                    # not hide an unrelated, plain "do not pay" source ban.
                    if candidate or not _is_conditional_ban(match.group(), after):
                        continue
                if match.group() in {"别", "別"} and (
                    not after.strip() or (before.strip() and not re.search(
                        r"(?:请|請|你|你们|你們|大家|千万|千萬|可|先)\s*$", before))
                ):
                    continue
                if re.search(r"(?:并非|並非|不是|并不是|並不是|并不|並不)\s*$", before):
                    continue
                if re.search(r"[?？]|(?:吗|嗎|么|麼|吧|呢)\s*$", clause):
                    continue
                if match.group() == "不要" and (after.startswith(("紧", "緊", "脸", "臉", "钱", "錢")) or
                                               re.search(r"(?:我|我们|我們)\s*$", before)):
                    continue
                if match.group() == "禁止":
                    if re.match(r"(?:并非|並非|不是|不成立|已经取消|已取消|已解除)", after):
                        continue
                    if re.search(r"(?:取消|解除)\s*$", before) or (not candidate and after.startswith(("不", "未", "没", "沒"))):
                        continue
                return True
        elif language == "ja":
            if not candidate and not re.search(r"[?？]|(?:ですか|ますか|ませんか|だろうか|でしょうか)\s*$", clause) and (
                _has_conjunctive_negative_request(clause) or _JA_STOP_REQUEST.search(clause)
            ):
                return True
            for match in _JA_PROHIBITION.finditer(clause):
                if match.group().startswith("ては") and clause[:match.start()].endswith("なく"):
                    # しなくてはならない expresses an obligation, not a ban.
                    continue
                if re.search(r"[?？]|(?:ですか|ますか|ませんか|だろうか|でしょうか)\s*$", clause):
                    continue
                if re.search(r"ないわけでは(?:ない|ありません)|ないとは限らない", clause):
                    continue
                after = clause[match.end():]
                if match.group() in {"禁止", "禁じ"} and re.match(
                    r"(?:では(?:ない|ありません)|じゃない|されて(?:いない|いません)|られて(?:いない|いません)|ではなく)", after
                ):
                    continue
                return True
    return False


def _conveys_conditional_prohibition(source: str, target: str, direction: str) -> bool:
    """Match negative-condition + negative-main-action request structures.

    This does not prove that arbitrary condition/action paraphrases are synonyms.
    A plain payment or send ban cannot be covered by an unrelated conditional.
    """
    if direction != "ja_to_zh":
        return False
    source_clauses = [clause for clause in _JA_CLAUSE_BREAK.split(source)
                      if _has_prohibition(clause, "ja")]
    if not source_clauses or not all(
        _has_conjunctive_negative_request(clause) and len(list(_JA_PROHIBITION.finditer(clause))) == 1
        for clause in source_clauses
    ):
        return False
    target_count = sum(
        _is_conditional_ban(match.group(), clause[match.end():])
        for clause in _CLAUSE_BREAK.split(target) if not re.search(r"[?？]", clause)
        for match in _ZH_TARGET_PROHIBITION.finditer(clause)
    )
    return target_count >= len(source_clauses)


def _conveys_indirect_japanese_prohibition(source: str, target: str, direction: str) -> bool:
    """Permit new stop/negative-converb forms only for supported same actions.

    An unrelated action left untouched does not preserve a ban on paying.
    The small action vocabulary deliberately leaves unknown actions and
    paraphrases unsupported instead of treating all やめて / ず as equivalents.
    """
    if direction != "zh_to_ja":
        return False
    source_clauses = [clause.strip() for clause in _CLAUSE_BREAK.split(source)
                      if _has_prohibition(clause, "zh")]
    target_clauses = [clause.strip() for clause in _JA_CLAUSE_BREAK.split(target)]
    ban_head = r"(?:请勿|請勿|不得|不准|不许|不許|不要|别|別)(?:再|先|暂时|暫時|马上|馬上|立即|擅自|随便|隨便)?"
    pairs = [
        (ban_head + r"(?:付款|支付|付钱|付錢|转账|轉賬|转帐|轉帳|汇款|匯款)",
         r"支払わず|払わず|送金せず|振り込まず|(?:お?支払い|送金|振り込み)(?:は|を)(?:やめて|止めて|おやめ)(?:ください|下さい)"),
        (ban_head + r"(?:移动|移動|动|動|碰|触碰|觸碰|触摸|觸摸|操作)",
         r"動かさず|動かず|触らず|触れず|操作せず|(?:移動|動かすこと|触ること|操作)(?:は|を)(?:やめて|止めて)(?:ください|下さい)"),
    ]
    return bool(source_clauses) and all(
        len(_ZH_PROHIBITION.findall(clause)) == 1 and any(
            re.search(zh_action, clause) and any(
                re.search(ja_action, candidate) and
                (_JA_STOP_REQUEST.search(candidate) or _has_conjunctive_negative_request(candidate))
                for candidate in target_clauses
            ) for zh_action, ja_action in pairs
        ) for clause in source_clauses
    )


def _preserves_forbidden_change(source: str, target: str, direction: str) -> bool:
    """Accept keeping fields unchanged only when every ban is a change ban.

    Distinct and explicitly coordinated source objects need enough distinct
    preservation objects. Existing amount/date distinctions catch obvious
    mismatches; other cross-language object identities cannot be proven here.
    The rule checks action and coverage, not a general bilingual object lexicon.
    """
    source_language, target_language = {"ja_to_zh": ("ja", "zh"), "zh_to_ja": ("zh", "ja")}[direction]
    ja_change = re.compile(
        r"(?:(?:変更|修正)し|変え)ないで(?:ください|下さい)|"
        r"(?:(?:変更|修正)して|変えて)は(?:いけない|いけません|ならない|なりません)|"
        r"(?:変更|修正)せず(?:に)?|変えず(?:に)?"
    )
    zh_change = re.compile(
        r"(?:请勿|請勿|不要|不得|不准|不许|不許|不可|别|別)(?:再|先|暂时|暫時)?"
        r"(?:修改|更改|变更|變更|调整|調整|改动|改動|改)"
    )
    modifiers = r"(?:请|請|先|暂时|暫時|目前|仍|务必|務必)"
    source_pattern = ja_change if source_language == "ja" else zh_change
    splitter = _JA_CLAUSE_BREAK if source_language == "ja" else _CLAUSE_BREAK
    source_objects = []
    for clause in splitter.split(source):
        if not _has_prohibition(clause, source_language):
            continue
        matches = list(source_pattern.finditer(clause))
        if not matches or _has_prohibition(source_pattern.sub("", clause), source_language):
            return False  # A payment/send/delete ban cannot become "unchanged".
        for match in matches:
            before = re.split(r"[、,，]", clause[:match.start()])[-1].strip()
            if source_language == "ja":
                subject = re.search(r"(.+?)(?:は|を|も)(?:まだ|まず|先に|当面|しばらく|どうか|絶対に)*$", before)
                value = subject[1] if subject else ""
            else:
                value = re.sub(modifiers + r"*$", "", before)
                if not value:
                    value = re.split(r"[、,，]", clause[match.end():])[0].strip()
            source_objects.extend(re.split(r"(?:と|や|及び|および|和|与|與|及|、)", value) if value else [f"#{len(source_objects)}"])
    if not source_objects:
        return False
    target_objects = []
    for clause in _CLAUSE_BREAK.split(target):
        if target_language == "zh":
            direct = re.search(r"(?:保持|维持|維持)(.+?)(?:不变|不變|原样|原樣|原状|原狀)", clause)
            subject = re.fullmatch(r"(.+?)(?:保持|维持|維持)(?:不变|不變|原样|原樣|原状|原狀)", clause.strip())
            value = direct[1] if direct else re.sub(modifiers + r"*$", "", subject[1]) if subject else ""
        else:
            subject = re.search(r"(.+?)(?:は|を)(?:そのまま|元のまま)に(?:してください|して下さい|保ってください)", clause)
            value = subject[1] if subject else ""
        if value:
            target_objects.extend(re.split(r"(?:と|や|及び|および|和|与|與|及|、)", value))
    source_objects = set(filter(None, (value.strip() for value in source_objects)))
    target_objects = set(filter(None, (value.strip() for value in target_objects)))
    if len(target_objects) < len(source_objects):
        return False
    known_families = (r"金額|料金|価格|代金|金额|数额|數額|价格|價格|费用|費用", r"日付|日期")
    return all(not any(re.search(family, value) for value in source_objects) or
               any(re.search(family, value) for value in target_objects) for family in known_families)


# Match a single, affirmative wish, not arbitrary uses of ない/不. The noisy
# いいないな tail is accepted only after the same affirmative conditional verb.
_JA_WISH_TAIL = r"(?:いいな(?:あ|ぁ)?|いいないな)(?:と思(?:う|います|ってい(?:る|ます)))?$"
_WISH_ACTIONS = tuple((re.compile(ja + _JA_WISH_TAIL), re.compile(zh)) for ja, zh in (
    (r"売(?:れたら|ったら)", r"(?:出售|售出|销售|銷售|售卖|售賣|出卖|出賣|卖|賣)"),
    (r"送(?:れたら|ったら)", r"(?:发送|發送|寄出|寄送|送出|送达|送達)"),
    (r"読(?:めたら|んだら)", r"(?:阅读|閱讀|读|讀)"),
    (r"参加できたら", r"参加|參加"),
))
_ZH_BAN_ADVERBS = re.compile(r"(?:再|先|暂时|暫時|马上|馬上|立即|继续|繼續|去|进行|進行)*")


def _reverses_explicit_japanese_wish(source: str, target: str) -> bool:
    """Reject only supported same-action wishes changed to a direct ban.

    Different actions, quoted/multiple clauses, negative wishes and questions
    remain outside this finite rule. For example, wanting rain to stop can
    legitimately be expressed as hoping it will not rain. Nothing is rewritten.
    """
    source = re.sub(r"[ \t\u3000]+", "", unicodedata.normalize("NFKC", source.strip()))
    source = source.rstrip("。.!！…")
    if re.search(r"[、,，。.!！?？;；\n\r「」『』“”\"'（）()]", source):
        return False
    for wish, action in _WISH_ACTIONS:
        match = wish.search(source)
        if match is None:
            continue
        # Prefix negation may have wider scope than the final wish. Prefer a
        # missed warning over guessing at that scope or repairing an ASR error.
        if re.search(r"ない|なく|なけれ|ません|ぬ|ず|まい|禁止|禁じ", source[:match.start()]):
            continue
        for clause in _CLAUSE_BREAK.split(target):
            markers = list(_ZH_TARGET_PROHIBITION.finditer(clause))
            if len(markers) != 1:
                continue
            marker = markers[0]
            is_ban = _has_prohibition(clause, "zh", candidate=True)
            if not is_ban and marker.group() in {"别", "別"} and re.fullmatch(
                r"(?:今天|今晚|明天|现在|現在|目前)\s*", clause[:marker.start()].strip()
            ):
                # An explicit time before 别 is a directive prefix; arbitrary
                # noun prefixes (识别/区别) must not become prohibitions.
                is_ban = _has_prohibition("请" + clause[marker.start():], "zh", candidate=True)
            if not is_ban:
                continue
            after = clause[marker.end():].lstrip()
            after = after[_ZH_BAN_ADVERBS.match(after).end():].lstrip()
            verb = action.match(after)
            if verb is not None and not re.match(
                r"(?:得|错|錯|太|不|给|給|向|到|成|为|為|便宜|低|高|快|慢|过多|過多)", after[verb.end():]
            ):
                # 不要卖得太便宜 / 不要读错 constrain manner or arguments;
                # they do not establish a ban on selling or reading itself.
                return True
    return False


def _japanese_counter_target_view(source: str, target: str) -> str:
    """Compare clear Arabic 2+ つ counts with Chinese kind counters only.

    This view is never used for source preparation or displayed text. Keeping
    it local avoids reclassifying lexical 一つ/一种 or corrupting 唯一一种.
    """
    current = unicodedata.normalize("NFKC", source)
    if not re.search(r"(?<![0-9A-Za-z_.])(?:[2-9]|[1-9][0-9]+)つ(?!目)", current):
        return target

    def replace(match):
        token = match[1]
        # Consecutive Han digits may be an identifier or lexical 一一, not a
        # clear multi-digit quantity. Explicit 十二/二十五 remain comparable.
        if len(token) > 1 and all(char in _DIGITS for char in token):
            return match.group()
        value = _parse_han_integer(token)
        if value is None or int(value) < 2:
            return match.group()
        return value + match[2]

    return re.sub(rf"(?<![{_HAN_NUMBER}0-9A-Za-z_点點.．])([{_HAN_NUMBER}]+)([种種])", replace, target)


def candidate_issues(source: str, target: str, direction: str) -> tuple[str, ...]:
    """Return stable quality, numeric, prohibition and wish error codes.

    The check deliberately misses ambiguous natural-language cases. A clean
    result does not establish that referents, intent, or the translation are right.
    When the current source has no explicit number, a candidate may resolve a
    referent to a number from context. This function receives no context and
    cannot verify whether an added number actually has that basis.
    """
    quality_issues = output_quality_issues(source, target)
    if quality_issues:
        return quality_issues
    issues = []
    positions = bool(_FIRST_POSITION.search(source) or _FIRST_POSITION.search(target))
    source_values = _numeric_values(source, per_units=_implicit_per_units(target), positions=positions)
    numeric_target = _japanese_counter_target_view(source, target) if direction == "ja_to_zh" else target
    if (source_values and
            not (_has_ambiguous_numeric_span(source) or _has_ambiguous_numeric_span(target)) and
            source_values != _numeric_values(numeric_target, per_units=_implicit_per_units(source), positions=positions)):
        issues.append("number_mismatch")
    languages = {"zh_to_ja": ("zh", "ja"), "ja_to_zh": ("ja", "zh")}.get(direction)
    if languages and _has_prohibition(source, languages[0]) and not _has_prohibition(
        target, languages[1], candidate=True
    ) and not _preserves_forbidden_change(source, target, direction) and not _conveys_indirect_japanese_prohibition(source, target, direction) and not _conveys_conditional_prohibition(source, target, direction):
        issues.append("prohibition")
    if direction == "ja_to_zh" and _reverses_explicit_japanese_wish(source, target):
        issues.append("intent_reversal")
    return tuple(issues)
