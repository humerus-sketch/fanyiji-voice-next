"""Local invented examples only; no model, runtime, or holdout data required."""

import pytest

from fanyiji_next.translation_guard import (
    candidate_issues, context_grounding_issues, output_quality_issues, prepare_source, source_fragment_issue,
)


@pytest.mark.parametrize("source,expected", [
    ("金额是一千零二十五元。", "金额是1025元。"),
    ("千二十五円を支払う。", "1025円を支払う。"),
    ("三点五美元，零点零五元。", "3.5美元，0.05元。"),
    ("二〇二六年十月十六日", "2026年10月16日"),
    ("一万二千三百四十五", "12345"),
    ("两百元和兩百円", "200元和200円"),
    ("壹佰贰拾伍元", "125元"),
    ("三人买了二件。", "3人买了2件。"),
    ("百分之三", "百分之3"),
    ("编号〇〇一", "编号001"),
    ("十", "10"),
    ("负三点五度", "负3.5度"),
    ("一緒に、一般情况下、一旦开始。", "一緒に、一般情况下、一旦开始。"),
    ("千万不要付款，十分注意安全。", "千万不要付款，十分注意安全。"),
    ("万一迟到，就一起走。", "万一迟到，就一起走。"),
    ("请逐一说明，一一确认。", "请逐一说明，一一确认。"),
    ("一百二元", "一百二元"),
    ("一万二円", "一万二円"),
    ("十百元", "十百元"),
    ("金额1万元", "金额1万元"),
    ("3.5千円", "3.5千円"),
    ("1万2千元", "1万2千元"),
    ("下午三点十五分", "下午3点15分"),
    ("三点五分钟", "3.5分钟"),
    ("三点十五", "三点十五"),
    ("一点儿也不喜欢，一点小事。", "一点儿也不喜欢，一点小事。"),
    ("下午一点整", "下午1点整"),
    ("もう一度確認してください。", "もう一度確認してください。"),
    ("もう一度だけ確認してください。", "もう1度だけ確認してください。"),
    ("再一次确认。", "再一次确认。"),
    ("请再确认一次日期。", "请再确认一次日期。"),
    ("请只再确认一次日期。", "请只再确认1次日期。"),
    ("二つ目は残して、一つ目も確認。", "2つ目は残して、1つ目も確認。"),
    ("二番目の項目", "2番目の項目"),
    ("五分待ってください。", "5分待ってください。"),
])
def test_prepare_source_only_normalizes_explicit_numeric_spans(source, expected):
    assert prepare_source(source) == expected


@pytest.mark.parametrize("source,target,direction", [
    ("千二十五円", "1025元", "ja_to_zh"),
    ("金额三点五元。", "金額は3.50元です。", "zh_to_ja"),
    ("二〇二六年十月十六日", "2026年10月16日", "zh_to_ja"),
    ("2026年01月02日", "二〇二六年一月二日", "zh_to_ja"),
    ("2026/01/02", "2026年1月2日", "zh_to_ja"),
    ("付款1,025元", "1025元を支払う", "zh_to_ja"),
    ("1万元", "一万円", "zh_to_ja"),
    ("3.5千円", "3500元", "ja_to_zh"),
    ("壹佰贰拾伍元", "125円", "zh_to_ja"),
    ("三点十五分出发", "3時15分に出発", "zh_to_ja"),
    ("再一次确认。", "もう一度確認してください。", "zh_to_ja"),
    ("もう一度だけ確認してください。", "请只再确认1次。", "ja_to_zh"),
    ("３．５０元", "三点五元", "ja_to_zh"),
    ("负三点五度", "マイナス3.5度", "zh_to_ja"),
    ("百分之三", "3%", "zh_to_ja"),
    ("10円と20円", "20元和10元", "ja_to_zh"),
    ("编号001", "番号001", "zh_to_ja"),
    ("不要付一百元。", "100元を支払わないでください。", "zh_to_ja"),
])
def test_equal_numbers_are_not_rejected(source, target, direction):
    assert candidate_issues(source, target, direction) == ()


@pytest.mark.parametrize("source,target,direction", [
    ("2つの方法がある", "有两种方法。", "ja_to_zh"),
    ("2つの方法がある", "有兩種方法。", "ja_to_zh"),
    ("2つの方法がある", "有二种方法。", "ja_to_zh"),
    ("3つの選択肢です。", "这是三种选项。", "ja_to_zh"),
    ("12つの種類です。", "有十二种。", "ja_to_zh"),
    ("25つです。", "有二十五種。", "ja_to_zh"),
    ("２つです。", "有两种。", "ja_to_zh"),
])
def test_arabic_japanese_counts_accept_clear_chinese_kind_counter_values(source, target, direction):
    assert candidate_issues(source, target, direction) == ()


@pytest.mark.parametrize("source,target,direction", [
    ("2つの方法がある", "有三种方法。", "ja_to_zh"),
    ("2つの方法がある", "有方法。", "ja_to_zh"),
    ("2つと2つです。", "有两种。", "ja_to_zh"),
    ("2つです。", "有两种和三种。", "ja_to_zh"),
    ("12つです。", "有二十一种。", "ja_to_zh"),
    ("2つです。", "唯一一种方法。", "ja_to_zh"),
    ("2つです。", "有一一种方法。", "ja_to_zh"),
    ("2つです。", "有两种，价格3元。", "ja_to_zh"),
])
def test_counter_equivalence_does_not_waive_changed_missing_or_repeated_values(source, target, direction):
    assert candidate_issues(source, target, direction) == ("number_mismatch",)


@pytest.mark.parametrize("source,target", [
    ("2.2つです。", "有二点二种。"),
    ("２．２つです。", "有二點二種。"),
    ("2.5つです。", "有二点五种。"),
    ("2つと2.2つです。", "有两种和二点二种。"),
    ("2つと２．２つです。", "有兩種和二點二種。"),
    ("2つと2.2つです。", "有二种和２．２种。"),
    ("2つと0.5つです。", "有两种和零点五种。"),
])
def test_target_counter_view_does_not_partially_rewrite_decimals_even_beside_an_integer_counter(source, target):
    assert candidate_issues(source, target, "ja_to_zh") == ()


@pytest.mark.parametrize("source,target", [
    ("2.2つです。", "有二点三种。"),
    ("２．２つです。", "有二點三種。"),
    ("2つと2.2つです。", "有两种和二点三种。"),
    ("2つと0.5つです。", "有两种和零点六种。"),
    ("2つと2.2つです。", "有二点二种。"),
])
def test_counter_decimal_boundary_abstention_does_not_waive_wrong_or_missing_values(source, target):
    assert candidate_issues(source, target, "ja_to_zh") == ("number_mismatch",)


@pytest.mark.parametrize("source", [
    "一緒に、一般情况下、一旦开始。", "同一性を確認。", "这是种子和种族的介绍。",
    "お前の大切な", "そうだろうな。", "唯一一种方法。", "二つの方法がある。",
    "一つお願いします。", "三つだけ残す。", "每一种方法都试试。", "两种方法。", "兩種方法。", "二种方法。",
])
def test_counter_suffixes_do_not_turn_unrelated_words_into_numerals(source):
    assert prepare_source(source) == source


@pytest.mark.parametrize("source,target,direction", [
    ("同一种方法。", "同じ方法。", "zh_to_ja"),
    ("同一種類です。", "相同种类。", "ja_to_zh"),
    ("一つ一つ確認してください。", "请逐一确认。", "ja_to_zh"),
    ("一つよろしくお願いします。", "请多关照。", "ja_to_zh"),
    ("一つずつ確認してください。", "请逐一确认。", "ja_to_zh"),
    ("一つずつ配ってください。", "请逐个分发。", "ja_to_zh"),
    ("一つ残らず回収する。", "全部收回。", "ja_to_zh"),
    ("何一つ分からない。", "什么都不懂。", "ja_to_zh"),
    ("一つには予算の問題だ。", "原因之一是预算问题。", "ja_to_zh"),
    ("一種の冗談だ。", "算是个玩笑。", "ja_to_zh"),
    ("一種独特な雰囲気だ。", "有种独特的氛围。", "ja_to_zh"),
    ("一つになって頑張ろう。", "团结起来努力吧。", "ja_to_zh"),
    ("每一种方法都试试。", "それぞれの方法を試して。", "zh_to_ja"),
    ("任何一种方法都可以。", "どの方法でもいい。", "zh_to_ja"),
    ("某一种方式。", "ある方法。", "zh_to_ja"),
    ("唯一一种方法。", "唯一の方法。", "zh_to_ja"),
    ("同样一种方法。", "同じ方法。", "zh_to_ja"),
    ("单一一种方案。", "単一の案です。", "zh_to_ja"),
    ("连一种方法都没有。", "方法が全くない。", "zh_to_ja"),
    ("只有一种方法。", "方法は一つだけです。", "zh_to_ja"),
    ("一つお願いします。", "请给我一个。", "ja_to_zh"),
])
def test_narrow_target_comparison_preserves_baseline_lexical_and_source_preparation(source, target, direction):
    assert candidate_issues(source, target, direction) == ()
    assert prepare_source(source) == source


@pytest.mark.parametrize("source,target,direction", [
    ("二つの方法がある。", "有三种方法。", "ja_to_zh"),
    ("三つだけ残す。", "只留两个。", "ja_to_zh"),
    ("只有一种方法。", "方法は二つだけです。", "zh_to_ja"),
])
def test_this_patch_does_not_expand_preexisting_unrecognized_han_counter_protection(source, target, direction):
    # These already bypassed numeric comparison in the baseline. Expanding
    # source semantics here caused unrelated false rejections, so keep scope.
    assert candidate_issues(source, target, direction) == ()
    assert prepare_source(source) == source


def test_existing_arabic_counts_still_protect_numbers_beside_lexical_one():
    assert candidate_issues("一つずつ確認して2個残す。", "请逐一确认，留下3个。", "ja_to_zh") == (
        "number_mismatch",)


@pytest.mark.parametrize("source,target", [
    ("千二十五円", "1250元"),
    ("金额三点五元", "金額は35元"),
    ("10元，10元", "10円"),
    ("10元", "10円と20円"),
    ("编号001", "番号1"),
    ("A001", "A1"),
    ("10元", "金額です"),
    ("负5度", "5度"),
    ("1万元", "1000円"),
])
def test_changed_missing_added_or_identifier_numbers_are_rejected(source, target):
    assert candidate_issues(source, target, "zh_to_ja") == ("number_mismatch",)


@pytest.mark.parametrize("source,target,direction", [
    ("请先不要付款。", "まず支払ってください。", "zh_to_ja"),
    ("禁止拍照。", "写真を撮ってください。", "zh_to_ja"),
    ("别打开门。", "ドアを開けてください。", "zh_to_ja"),
    ("支払わないでください。", "请付款。", "ja_to_zh"),
    ("ここでは撮影してはいけません。", "这里可以拍照。", "ja_to_zh"),
    ("触るな。", "请触摸。", "ja_to_zh"),
])
def test_missing_explicit_prohibition_is_rejected(source, target, direction):
    assert candidate_issues(source, target, direction) == ("prohibition",)


@pytest.mark.parametrize("source,target", [
    ("そうだろうな", "大概是吧。"),
    ("そうだろうな。", "想也是这样。"),
    ("そうでしょうな。", "应该是这样吧。"),
    ("そうだろうな！", "大概是这样啊！"),
    ("彼もそう考えるだろうな。", "他大概也会这么想吧。"),
])
def test_conjectural_darouna_and_deshouna_do_not_require_a_prohibition(source, target):
    assert candidate_issues(source, target, "ja_to_zh") == ()


@pytest.mark.parametrize("source,target", [
    ("行くな。", "请去。"),
    ("するな。", "请做。"),
    ("送るな。", "请发送。"),
    ("読むな。", "请阅读。"),
    ("言うな。", "请说出来。"),
    ("そう思うな。", "请那样想。"),
    ("行くな。そうだろうな。", "大概是吧。"),
    ("そうでしょうな、送るな。", "应该是这样吧。"),
    ("送らないでください、そうでしょうな。", "应该是这样吧。"),
])
def test_real_verb_na_bans_remain_protected_alongside_conjectural_tails(source, target):
    assert candidate_issues(source, target, "ja_to_zh") == ("prohibition",)


@pytest.mark.parametrize("source,target", [
    ("请不要去。", "行くだろうな。"),
    ("禁止阅读。", "読むだろうな。"),
    ("不要做。", "そうでしょうな。"),
])
def test_conjectural_japanese_candidate_cannot_mask_a_real_source_ban(source, target):
    assert candidate_issues(source, target, "zh_to_ja") == ("prohibition",)


@pytest.mark.parametrize("source,target,direction", [
    ("请不要付款。", "支払わないでください。", "zh_to_ja"),
    ("不得付款。", "お支払いはお控えください。", "zh_to_ja"),
    ("禁止拍照。", "撮影は禁止です。", "zh_to_ja"),
    ("支払わないでください。", "请勿付款。", "ja_to_zh"),
    ("这里不是合同，而是车票。", "ここでは契約を確認してください。", "zh_to_ja"),
    ("并非不能付款。", "支払ってください。", "zh_to_ja"),
    ("不能不付款。", "支払ってください。", "zh_to_ja"),
    ("不是不要付款。", "支払ってください。", "zh_to_ja"),
    ("为什么不要付款？", "支払ってください。", "zh_to_ja"),
    ("不要付款吧。", "支払ってください。", "zh_to_ja"),
    ("我不要这个。", "これをください。", "zh_to_ja"),
    ("不要紧。", "大丈夫です。", "zh_to_ja"),
    ("特别好。", "いいです。", "zh_to_ja"),
    ("这是区别。", "これは違いです。", "zh_to_ja"),
    ("分别识别级别。", "識別します。", "zh_to_ja"),
    ("禁止不付款。", "支払ってください。", "zh_to_ja"),
    ("解除禁止。", "利用できます。", "zh_to_ja"),
    ("支払わないわけではない。", "请付款。", "ja_to_zh"),
    ("支払わないほうがいい。", "请付款。", "ja_to_zh"),
    ("支払わないでくださいませんか？", "请付款。", "ja_to_zh"),
    ("禁止ではありません。", "可以。", "ja_to_zh"),
])
def test_retained_prohibition_or_ambiguous_negatives_are_not_rejected(source, target, direction):
    assert candidate_issues(source, target, direction) == ()


def test_contract_negation_cannot_mask_lost_payment_prohibition():
    source = "这次说的是车票，不是前面的合同，请先不要付款。"
    prefix = "今回はチケットのことです。前の契約ではありませんので、"
    assert candidate_issues(source, prefix + "まずお支払いをお願いします。", "zh_to_ja") == ("prohibition",)
    assert candidate_issues(source, prefix + "まずはお支払いしないでください。", "zh_to_ja") == ()


def test_issue_order_is_stable_and_target_is_never_repaired():
    target = "1250円を支払ってください。"
    assert candidate_issues("一千零二十五元，请勿付款。", target, "zh_to_ja") == (
        "number_mismatch", "prohibition")
    assert target == "1250円を支払ってください。"


def test_unsupported_direction_only_checks_numbers():
    assert candidate_issues("不要付10元。", "Pay 20.", "unsupported") == ("number_mismatch",)


@pytest.mark.parametrize("source", [
    "我不得不付款。", "不许不付款。", "不准不付款。", "请勿不付款。", "不要不回复。",
    "不要 不回复。",
])
def test_double_negative_directives_do_not_require_a_prohibition(source):
    assert candidate_issues(source, "支払ってください。", "zh_to_ja") == ()


@pytest.mark.parametrize("target", [
    "我不得不付款。", "不许不付款。", "不准不付款。", "请勿不付款。", "不要不付款。",
    "不能不付款。",
])
def test_double_negative_candidate_cannot_mask_a_lost_prohibition(target):
    assert candidate_issues("支払わないでください。", target, "ja_to_zh") == ("prohibition",)


def test_japanese_double_negative_obligation_is_not_a_prohibition_in_either_role():
    obligation = "支払わなくてはならない。"
    assert candidate_issues(obligation, "必须付款。", "ja_to_zh") == ()
    assert candidate_issues("不要付款。", obligation, "zh_to_ja") == ("prohibition",)


@pytest.mark.parametrize("source,target", [
    ("一百二元", "120円"),
    ("一万二円", "12000元"),
    ("1万2千元", "12000円"),
    ("三分之一", "1/3"),
    ("三分の一", "1/3"),
    ("三点十五", "3時15分"),
])
def test_ambiguous_or_unsupported_mixed_numerals_do_not_claim_a_mismatch(source, target):
    assert candidate_issues(source, target, "zh_to_ja") == ()


@pytest.mark.parametrize("source,target", [
    ("部長、日付だけもう一度確認してください。金額はまだ変更しないでください。",
     "部长，请再次确认日期。请暂时不要更改金额。"),
    ("二つ目は残して、一つ目もまだ削除しないでください。",
     "第二个保留下来，第一个也不要删除。"),
    ("彼がまだ来ていなければ五分待って、着いてから始めましょう。",
     "如果他还没来，请等五分钟，等他到了再开始吧。"),
])
def test_development_regressions_for_frequency_ordinals_and_minutes(source, target):
    assert candidate_issues(source, target, "ja_to_zh") == ()


def test_explicit_ordinal_and_minute_changes_still_fail():
    assert candidate_issues("二つ目を残してください。", "请保留第3个。", "ja_to_zh") == ("number_mismatch",)
    assert candidate_issues("五分待ってください。", "请等6分钟。", "ja_to_zh") == ("number_mismatch",)
    assert candidate_issues("もう一度だけ確認。", "只确认2次。", "ja_to_zh") == ("number_mismatch",)


def test_numberless_referent_may_resolve_to_amount_without_proving_context_basis():
    source = "请把它改成上文约定的金额。"
    target = "前述の金額である1250円に変更してください。"
    # No context is passed here: acceptance permits reference resolution but
    # does not establish that 1250 was mentioned in the preceding conversation.
    assert candidate_issues(source, target, "zh_to_ja") == ()


def test_current_explicit_amount_stays_protected_despite_a_referent():
    source = "目前金额是1025元，请保持这个金额。"
    target = "現在の金額は1250元です。その金額を維持してください。"
    assert candidate_issues(source, target, "zh_to_ja") == ("number_mismatch",)


def test_repeated_check_equivalence_is_symmetric_with_other_explicit_values():
    source = "主任，麻烦您再确认一次日期，金额15元先不要改。"
    target = "主任、お願いですが、日付をもう一度確認してください。15元の金額は変更しないでください。"
    assert candidate_issues(source, target, "zh_to_ja") == ()
    assert candidate_issues("请只再确认一次日期。", "日付を2度確認してください。", "zh_to_ja") == ("number_mismatch",)


def test_development_repeated_check_and_unchanged_amount_regressions():
    assert candidate_issues(
        "主任，麻烦您再确认一次日期，金额先不要改。",
        "主任、お願いですが、日付をもう一度確認してください。金額は変更しないでください。",
        "zh_to_ja") == ()
    assert candidate_issues(
        "部長、日付だけもう一度確認してください。金額はまだ変更しないでください。",
        "部长，请再次确认日期。金额请保持不变。", "ja_to_zh") == ()


@pytest.mark.parametrize("source,target,direction", [
    ("金額を変更しないでください。", "请保持金额不变。", "ja_to_zh"),
    ("日付はまだ変更しないでください。", "日期先保持不变。", "ja_to_zh"),
    ("金额先不要改。", "金額はそのままにしてください。", "zh_to_ja"),
    ("日期暂时不要修改。", "日付はそのままにしてください。", "zh_to_ja"),
])
def test_same_object_no_change_and_keep_unchanged_are_narrow_equivalents(source, target, direction):
    assert candidate_issues(source, target, direction) == ()


@pytest.mark.parametrize("source,target", [
    ("支払わないでください。", "金额请保持不变。"),
    ("金額は変更しないでください。", "日期请保持不变。"),
    ("金額は変更しないでください。", "态度保持不变，请更改金额。"),
    ("金額は変更しないでください。支払わないでください。", "金额请保持不变。请付款。"),
    ("金額は変更しないでください。日付は変更しないでください。", "金额请保持不变。"),
])
def test_unrelated_unchanged_text_cannot_mask_an_uncovered_prohibition(source, target):
    assert candidate_issues(source, target, "ja_to_zh") == ("prohibition",)


@pytest.mark.parametrize("source,expected", [
    ("编号是零零八一六，不要写成八一六。", "编号是00816，不要写成816。"),
    ("请保留两个零。", "请保留2个0。"),
    ("0.08百分比", "0.08百分比"),
    ("百分比很低。", "百分比很低。"),
    ("四時半", "4時半"),
    ("每人最多两张。", "每人最多2张。"),
])
def test_numeric_lexical_contexts_preserve_identifiers_and_units(source, expected):
    assert prepare_source(source) == expected


@pytest.mark.parametrize("source,target,direction", [
    ("每人最多两张，儿童也算一人。", "1人あたり最大2枚、子供も1人です。", "zh_to_ja"),
    ("各组领取两件。", "1組あたり2件を受け取ってください。", "zh_to_ja"),
    ("每户两件。", "1世帯あたり2件です。", "zh_to_ja"),
    ("一人当たり二枚です。", "每人两张。", "ja_to_zh"),
    ("编号是零零八一六，保留两个零，不要写成八一六。",
     "番号は00816です。2つの0を残してください。816と書かないでください。", "zh_to_ja"),
    ("保留两个零。", "2つの0を残してください。", "zh_to_ja"),
    ("0.08パーセントです。", "0.08百分比。", "ja_to_zh"),
    ("百分之零点零八", "0.08%", "zh_to_ja"),
    ("四時半です。", "4点30分。", "ja_to_zh"),
    ("下午四点半。", "午後4時30分です。", "zh_to_ja"),
])
def test_distributive_one_zero_names_percent_and_half_hours_are_equivalent(source, target, direction):
    assert candidate_issues(source, target, direction) == ()


@pytest.mark.parametrize("source,target,direction", [
    ("每人最多两张。", "2人あたり2枚です。", "zh_to_ja"),
    ("每人最多一张。", "1人あたり2枚です。", "zh_to_ja"),
    ("每人最多两张，儿童算一人。", "1人あたり2枚、子供は2人です。", "zh_to_ja"),
    ("每人最多两张。", "1組あたり2枚です。", "zh_to_ja"),
    ("2人领取2张。", "1人あたり2枚です。", "zh_to_ja"),
    ("编号零零八一六", "番号816", "zh_to_ja"),
    ("保留两个零。", "2つの1を残してください。", "zh_to_ja"),
    ("0.08パーセントです。", "0.8百分比。", "ja_to_zh"),
    ("四時半です。", "4点45分。", "ja_to_zh"),
])
def test_numeric_context_equivalences_do_not_waive_wrong_values(source, target, direction):
    assert candidate_issues(source, target, direction) == ("number_mismatch",)


def test_choice_question_survives_a_comma_before_its_question_mark():
    assert candidate_issues("要不要先开窗，再开始播放？", "窓を開けてから再生した方がいいですか？", "zh_to_ja") == ()
    assert candidate_issues("能不能先不要付款，再讨论？", "先に相談できますか？", "zh_to_ja") == ()
    assert candidate_issues("不要付款，你明白了吗？", "支払ってください。", "zh_to_ja") == ("prohibition",)


def test_an_explicitly_revoked_instruction_is_not_a_current_prohibition():
    assert candidate_issues("之前让你不要启动的命令已经取消了，现在可以启动。",
                            "以前の停止指示は取り消されました。今は起動して構いません。", "zh_to_ja") == ()
    assert candidate_issues("之前让你不要付款的命令没有取消。", "支払ってください。", "zh_to_ja") == ("prohibition",)
    assert candidate_issues("之前让你不要付款的命令已经取消了，现在请不要付款。",
                            "今は支払ってください。", "zh_to_ja") == ("prohibition",)


def test_japanese_negative_converb_request_retains_its_prohibition():
    source = "荷物はまだ送らず、担当者が着いてから渡してください。"
    assert candidate_issues(source, "还没有寄出货物，等负责人到了再交给他。", "ja_to_zh") == ("prohibition",)
    assert candidate_issues(source, "先不要寄出货物，等负责人到了再交给他。", "ja_to_zh") == ()


@pytest.mark.parametrize("source", [
    "送らずに済んだので、安心してください。", "必ず確認してください。", "まず支払ってください。",
    "少しずつ支払ってください。", "いずれ来てください。", "絶えず注意してください。",
    "迷わず申し込んでください。", "限らず確認してください。",
])
def test_descriptions_and_lexical_zu_expressions_do_not_create_prohibitions(source):
    assert candidate_issues(source, "请确认。", "ja_to_zh") == ()


@pytest.mark.parametrize("source,target", [
    ("请不要付款。", "支払わず、窓口で確認してください。"),
    ("请不要付款。", "支払いをやめてください。"),
    ("先别移动箱子。", "箱は動かさず、位置を確認してください。"),
    ("不要操作按钮。", "操作をやめてください。"),
])
def test_new_indirect_japanese_prohibitions_require_a_supported_source_action(source, target):
    assert candidate_issues(source, target, "zh_to_ja") == ()


@pytest.mark.parametrize("target", [
    "金額はそのままにしてください。", "移動をやめてください。", "やめてください。",
    "部屋は動かさず、先に支払ってください。", "前の機種はやめてください。支払ってください。",
])
def test_unchanged_stop_or_zu_for_another_action_cannot_mask_do_not_pay(target):
    assert candidate_issues("请不要付款。", target, "zh_to_ja") == ("prohibition",)


def test_an_action_mentioned_only_in_an_object_is_not_the_prohibited_action():
    assert candidate_issues("不要移动付款设备。", "支払いをやめてください。", "zh_to_ja") == ("prohibition",)
    assert candidate_issues("不要付款，也不要动桌子。", "机は動かさず、先に支払ってください。", "zh_to_ja") == ("prohibition",)


@pytest.mark.parametrize("source,target", [
    ("料金はまだ変えないでください。", "价格请保持不变。"),
    ("価格を変更しないでください。", "请保持价格不变。"),
    ("代金は変更しないでください。", "费用先保持不变。"),
])
def test_explicit_fee_price_no_change_uses_the_narrow_value_object_family(source, target):
    assert candidate_issues(source, target, "ja_to_zh") == ()


def test_unknown_action_or_unrelated_unchanged_object_stays_outside_the_equivalence():
    assert candidate_issues("料金は変えないでください。", "日期保持不变，价格请更改。", "ja_to_zh") == ("prohibition",)
    assert candidate_issues("不要再替换成之前那种。", "前のものはやめてください。", "zh_to_ja") == ("prohibition",)


@pytest.mark.parametrize("source,target,direction", [
    ("次は箱の幅です。重さの欄は変えず、幅だけ測り直してください。",
     "接下来是箱子的宽度。保持重量栏不变，只需重新测量宽度即可。", "ja_to_zh"),
    ("係数の欄は変えず、右の欄だけ確認してください。",
     "保持系数栏不变，只检查右边一栏。", "ja_to_zh"),
    ("ラベルの色は変更しないでください。", "请保持标签颜色不变。", "ja_to_zh"),
    ("布局先不要改。", "配置はそのままにしてください。", "zh_to_ja"),
    ("幅と高さは変えないでください。", "请保持宽度和高度不变。", "ja_to_zh"),
])
def test_keep_unchanged_generalizes_only_the_change_action_without_an_object_dictionary(source, target, direction):
    assert candidate_issues(source, target, direction) == ()


@pytest.mark.parametrize("source,target,direction", [
    ("幅は変えず、荷物は送らず、担当者に確認してください。",
     "保持宽度不变，请确认后寄出货物。", "ja_to_zh"),
    ("宽度不要改，也不要付款。", "幅はそのままにしてください。", "zh_to_ja"),
    ("幅と高さは変えないでください。", "请保持宽度不变。", "ja_to_zh"),
    ("幅は変えず、高さも変えず、そのままにしてください。", "请保持宽度不变。", "ja_to_zh"),
])
def test_general_unchanged_equivalence_cannot_cover_other_bans_or_only_one_of_multiple_fields(source, target, direction):
    assert candidate_issues(source, target, direction) == ("prohibition",)


@pytest.mark.parametrize("source,expected", [
    ("三个零件长三点六毫米。", "3个零件长3.6毫米。"),
    ("零售店有两个零件。", "零售店有2个零件。"),
    ("零食和零工。", "零食和零工。"),
    ("三个零也要保留。", "3个0也要保留。"),
    ("请保留三个“零”字符。", "请保留3个“0”字符。"),
    ("数字零。", "数字0。"),
    ("零件数量为零。", "零件数量为0。"),
    ("编号是零。", "编号是0。"),
    ("零元和零点五元。", "0元和0.5元。"),
])
def test_zero_is_a_number_only_in_numeric_or_literal_digit_contexts(source, expected):
    assert prepare_source(source) == expected


@pytest.mark.parametrize("source,target,direction", [
    ("部品の長さは3.5ミリです。35ミリではありません。", "零件长度是3.5毫米，不是35毫米。", "ja_to_zh"),
    ("部品は二個で3.5ミリです。", "两个零件长3.5毫米。", "ja_to_zh"),
    ("每张三十元。", "1枚あたり30元です。", "zh_to_ja"),
    ("各张三十元。", "1枚につき30元です。", "zh_to_ja"),
    ("一枚当たり三十円です。", "每张30元。", "ja_to_zh"),
    ("每箱装十二件。", "1箱あたり12件です。", "zh_to_ja"),
    ("一箱に入れる数は十二個です。", "每箱装12个。", "ja_to_zh"),
    ("追加するのは四箱で、一箱に入れる数は十二個です。", "追加4箱，每箱装12个。", "ja_to_zh"),
    ("每瓶两件。", "1ボトルあたり2件です。", "zh_to_ja"),
])
def test_parts_and_common_unit_denominators_do_not_add_spurious_numbers(source, target, direction):
    assert candidate_issues(source, target, direction) == ()


@pytest.mark.parametrize("source,target,direction", [
    ("部品の長さは3.5ミリです。", "零件长度是35毫米。", "ja_to_zh"),
    ("每张三十元。", "2枚あたり30元です。", "zh_to_ja"),
    ("每张三十元。", "1箱あたり30元です。", "zh_to_ja"),
    ("每张三十元。", "1票あたり30元です。", "zh_to_ja"),
    ("每箱一个。", "1箱あたり2個です。", "zh_to_ja"),
    ("一箱に入れる数は十二個です。", "每箱装13个。", "ja_to_zh"),
    ("追加するのは一箱で、内容は十二個です。", "每箱装12个。", "ja_to_zh"),
])
def test_denominator_equivalence_preserves_the_actual_unit_quantity_and_amount(source, target, direction):
    assert candidate_issues(source, target, direction) == ("number_mismatch",)


@pytest.mark.parametrize("source,target,direction", [
    ("最初に三百円、残りは二百円です。", "第一次300元，剩余200元。", "ja_to_zh"),
    ("最初に二回確認してから三百円を払います。", "第一次先确认2次，再付300元。", "ja_to_zh"),
    ("第一次支付三百元。", "最初に300元を支払います。", "zh_to_ja"),
    ("最初に三百円です。", "首先300元。", "ja_to_zh"),
])
def test_lexical_first_and_first_sequence_position_are_equivalent(source, target, direction):
    assert candidate_issues(source, target, direction) == ()


@pytest.mark.parametrize("source,target", [
    ("最初に三百円です。", "第二次300元。"),
    ("最初に二回確認します。", "第一次确认3次。"),
    ("最初に一円です。", "第一次2元。"),
    ("最初に実行してください。", "请第二次执行。"),
])
def test_first_position_cannot_consume_a_real_amount_or_changed_count(source, target):
    assert candidate_issues(source, target, "ja_to_zh") == ("number_mismatch",)


@pytest.mark.parametrize("source,target,direction", [
    ("やっぱり難易度は下げるけど、最初からはやり直さない。", "不过还是决定降低难度，但不会从一开始就重新开始。", "ja_to_zh"),
    ("最初からやり直す。", "从头再来。", "ja_to_zh"),
    ("最初から二回確認してください。", "请从头确认2次。", "ja_to_zh"),
    ("请从头确认2次。", "最初から2回確認してください。", "zh_to_ja"),
    ("最初から300円を払うつもりでした。", "一开始打算支付300元。", "ja_to_zh"),
    ("首次支付300元。", "第1回に300元を支払います。", "zh_to_ja"),
    ("第一回支付300元。", "初回に300元を支払います。", "zh_to_ja"),
    ("初回は300円です。", "首次是300元。", "ja_to_zh"),
    ("首回は300円です。", "第一回是300元。", "ja_to_zh"),
])
def test_from_beginning_is_not_an_ordinal_but_actual_first_attempts_remain_equivalent(source, target, direction):
    assert candidate_issues(source, target, direction) == ()


@pytest.mark.parametrize("source,target,direction", [
    ("最初から二回確認してください。", "请从头确认3次。", "ja_to_zh"),
    ("最初から300円を払うつもりでした。", "一开始打算支付3元。", "ja_to_zh"),
    ("首次执行。", "第2回に実行します。", "zh_to_ja"),
    ("第一回支付300元。", "第2回に300元を支払います。", "zh_to_ja"),
    ("初回は300円です。", "第二次是300元。", "ja_to_zh"),
    ("首次确认2次，再付款300元。", "初回は3回確認してから300元を支払います。", "zh_to_ja"),
    ("首次支付一元。", "初回は2元を支払います。", "zh_to_ja"),
])
def test_beginning_abstention_does_not_mask_changed_counts_amounts_or_first_attempts(source, target, direction):
    assert candidate_issues(source, target, direction) == ("number_mismatch",)


@pytest.mark.parametrize("source,target,direction", [
    ("相談せずに決めないでください。", "请不要不商量就做决定。", "ja_to_zh"),
    ("不要不确认就签字。", "確認せずに署名しないでください。", "zh_to_ja"),
])
def test_explicit_conditional_ban_is_not_erased_as_a_simple_double_negative(source, target, direction):
    assert candidate_issues(source, target, direction) == ()


@pytest.mark.parametrize("source,target,direction", [
    ("不要没付款就离开。", "支払ってください。", "zh_to_ja"),
    ("请勿未付款就离开。", "支払ってください。", "zh_to_ja"),
    ("相談せずに決めないでください。", "请直接决定。", "ja_to_zh"),
    ("支払わないでください。", "不要不付款就离开。", "ja_to_zh"),
    ("支払わないでください。", "请不要不商量就决定。", "ja_to_zh"),
    ("送らないでください。", "请不要不商量就发送。", "ja_to_zh"),
    ("まず支払わないでください。", "请不要不商量就决定。", "ja_to_zh"),
    ("相談せずに決めないでください。支払わないでください。", "请不要不商量就决定。", "ja_to_zh"),
])
def test_a_conditional_ban_cannot_mask_an_unrelated_plain_ban(source, target, direction):
    assert candidate_issues(source, target, direction) == ("prohibition",)


@pytest.mark.parametrize("source,target", [
    ("売れたらいいな。", "不要出售。"),
    ("売れたらいいないな。", "不要出售。"),
    ("この絵が売れたらいいなあ。", "这幅画请不要再出售。"),
    ("古い本が売れたらいいなと思います。", "禁止售卖旧书。"),
    ("荷物を送れたらいいな。", "请勿寄出货物。"),
    ("今夜この本を読めたらいいなぁ。", "今晚别读这本书。"),
    ("会議に参加できたらいいな。", "不能参加会议。"),
])
def test_supported_affirmative_wish_cannot_become_a_same_action_ban(source, target):
    assert candidate_issues(source, target, "ja_to_zh") == ("intent_reversal",)


@pytest.mark.parametrize("source,target", [
    ("売れたらいいな。", "要是能卖出去就好了。"),
    ("売れたらいいないな。", "希望能卖出去。"),
    ("荷物を送れたらいいな。", "希望能寄出货物。"),
    ("今夜この本を読めたらいいな。", "希望今晚能读这本书。"),
    ("会議に参加できたらいいな。", "希望能参加会议。"),
    ("売らなければいいな。", "希望不要出售。"),
    ("売れないといいな。", "希望卖不出去。"),
    ("売りたくない。", "我不想出售。"),
    ("売らないでください。", "请不要出售。"),
    ("売ってもいいな。", "可以出售。"),
    ("売らなくてもいいな。", "可以不出售。"),
    ("売らなくてはならない。", "必须出售。"),
    ("売らないでくださいませんか？", "能否请您不要出售？"),
    ("売れたらいいなと思わない？", "你不希望能卖出去吗？"),
    ("売れたらいいな。", "要不要出售？"),
    ("売れたらいいな。", "并不是不要出售。"),
    ("売れたらいいな。", "不得不出售。"),
    ("売れたらいいな。", "不要不出售。"),
    ("売れたらいいな。", "不要忘记出售。"),
    ("売れたらいいな。", "不要出售的命令已经取消。"),
    ("売れたらいいな。", "不要出售吧。"),
    ("今夜この本を読めたらいいな。", "智能识别读书行为。"),
    ("今夜この本を読めたらいいな。", "今晚别读吧。"),
    ("今夜この本を読めたらいいな。", "今晚别读的命令已经取消。"),
    ("高く売れたらいいな。", "不要卖得太便宜就好了。"),
    ("高く売れたらいいな。", "不要卖便宜了。"),
    ("正しく読めたらいいな。", "希望不要读错。"),
    ("ゆっくり読めたらいいな。", "不要读得太快就好了。"),
    ("適度に参加できたらいいな。", "不要参加太多活动就好了。"),
    ("雨が止んだらいいな。", "希望不要再下雨了。"),
    ("騒音がなくなったらいいな。", "不要再有噪音就好了。"),
    ("売れたらいいな。", "希望能卖出去，不要寄出货物。"),
    ("売れたらいいな、でも今日は送らないでください。", "希望能卖出去，今天不要寄出。"),
    ("「売れたらいいな」とは思っていない。", "并不希望能卖出去。"),
    ("「売れたらいいな」と言わないでください。", "不要说希望能卖出去。"),
    ("誰にも知られず売れたらいいな。", "希望能悄悄卖出去。"),
])
def test_wish_guard_preserves_valid_negatives_permissions_questions_and_scope(source, target):
    assert candidate_issues(source, target, "ja_to_zh") == ()


def test_wish_guard_is_direction_limited_and_never_rewrites_the_candidate():
    target = "不要出售。"
    assert candidate_issues("売れたらいいな。", target, "ja_to_zh") == ("intent_reversal",)
    assert candidate_issues("売れたらいいな。", target, "zh_to_ja") == ()
    assert candidate_issues("売れたらいいな。", target, "unsupported") == ()
    assert target == "不要出售。"


def test_wish_issue_follows_numeric_reason_in_stable_order():
    assert candidate_issues("100円で売れたらいいな。", "不要卖200元。", "ja_to_zh") == (
        "number_mismatch", "intent_reversal")


@pytest.mark.parametrize("source,expected", [
    ("今回は十二・五グラムです。", "今回は12.5グラムです。"),
    ("二・五キログラム", "2.5キログラム"),
    ("零・〇五ミリグラム", "0.05ミリグラム"),
    ("長さは三・五センチメートルです。", "長さは3.5センチメートルです。"),
    ("一・二五リットル", "1.25リットル"),
    ("十二・五パーセント", "12.5パーセント"),
    ("12・50グラム", "12.50グラム"),
    ("１２･５０グラム", "12.50グラム"),
    ("十二・五・六グラム", "十二・五・六グラム"),
    ("二・五グラム・七グラム", "二・五グラム・七グラム"),
    ("一・二・五グラム", "一・二・五グラム"),
    ("二・五", "二・五"),
    ("一・五郎という名前", "一・五郎という名前"),
    ("二・五グラムフォン", "二・五グラムフォン"),
    ("型番A12・5グラム", "型番A12・5グラム"),
    ("商品名は「二・五グラム」です。", "商品名は「二・五グラム」です。"),
    ("商品名は「十二・五グラム」です。", "商品名は「十二・五グラム」です。"),
    ('商品名は"二・五グラム"です。', '商品名は"二・五グラム"です。'),
    ("一万二・五グラム", "一万二・五グラム"),
    ("二・十五グラム", "二・十五グラム"),
])
def test_japanese_middle_dot_decimal_needs_an_unambiguous_measurement(source, expected):
    assert prepare_source(source) == expected
    assert prepare_source(expected) == expected


def test_posthoc_japanese_middle_dot_weight_regression_is_not_a_number_mismatch():
    # This sentence was inspected before this repair; it is a seen regression.
    source = "今回は十二・五グラムで、百二十五グラムじゃないよ。"
    assert prepare_source(source) == "今回は12.5グラムで、125グラムじゃないよ。"
    assert candidate_issues(source, "这次是12.5克，不是125克。", "ja_to_zh") == ()


@pytest.mark.parametrize("source,target,expected", [
    ("十二・五グラム", "12.5克", ()),
    ("二・五リットル", "2.5升", ()),
    ("12・50グラム", "12.5克", ()),
    ("十二・五グラム", "125克", ("number_mismatch",)),
    ("十二・五グラム", "1.25克", ("number_mismatch",)),
    ("十二・五グラム", "12克和5克", ("number_mismatch",)),
    ("十二・五グラム", "12.6克", ("number_mismatch",)),
    ("一・二・五グラム", "1.25克", ()),
    ("二・五", "2.5", ()),
    ("商品名は「二・五グラム」です。", "商品名是二五克。", ()),
])
def test_middle_dot_measurement_comparison_preserves_values_and_abstains_on_ambiguity(source, target, expected):
    assert candidate_issues(source, target, "ja_to_zh") == expected


@pytest.mark.parametrize("source", [
    "たったった!", "タッタッタッ！", "たっ、たっ、たっ！", "だっだっだっ", "てってって",
    "しーしー", "シー、シー！", "ｼｰｼｰ", "しーっ、しーっ", "たったった！たったった！",
])
def test_isolated_repeated_beats_retain_uncertain_source_semantics(source):
    assert source_fragment_issue(source, "ja_to_zh") == "unresolved_phonetic_fragment"
    assert source_fragment_issue(source, "zh_to_ja") is None
    assert source_fragment_issue(source, "unsupported") is None


@pytest.mark.parametrize("source", [
    "ナイス！", "やった！", "やった、やった！", "はいはい。", "いやいや。", "そうそう。",
    "いいね！", "ちょっと！", "だめ。", "たった二人です。", "とっとと帰って。", "とっとと！", "ねーねー！",
    "ヨーヨー。", "はっはっは！", "シーシーさんも来る？", "シーシーは名前です。",
    "「しーしー」と言った。", "タッタッタって歌って。", "リリ、待って。", "しー。",
    "スースー。", "たった。", "きらきら！", "不要付款。", "", "！",
    "たたかう", "たった", "あたたかい", "あたたた", "ハイハイ", "やったやった",
])
def test_fragment_rule_does_not_treat_kana_shortness_names_or_real_repetition_as_noise(source):
    assert source_fragment_issue(source, "ja_to_zh") is None


@pytest.mark.parametrize("source,target,expected", [
    ("私の心そっと覗いてみませんか 欲し。", "想" * 15, ("output_repetition",)),
    ("私の心そっと覗いてみませんか 欲し。", "想" * 15 + "。", ("output_repetition",)),
    ("私の心そっと覗いてみませんか。", "想要看看" * 8, ("output_repetition",)),
    ("私の心そっと覗いてみませんか。", "想，" * 12, ("output_repetition",)),
    ("普通的一句话。", "心愿" * 6 + "的问题", ("output_repetition",)),
    ("普通的一句话。", "はい" * 8, ("output_repetition",)),
    ("普通的一句话。", "?" * 100, ("output_punctuation",)),
    ("普通的一句话。", "？？！" * 8, ("output_punctuation",)),
    ("普通的一句话。", "\ufffd" * 15, ("output_invalid_characters",)),
    ("普通的一句话。", "\ufffd？" * 12, ("output_invalid_characters",)),
    ("普通的一句话。", "这" + "\ufffd" * 15, ("output_invalid_characters",)),
    ("冴えない私だけど。", "⁇  ⁇  ⁇ , ⁇  ⁇ 更 ⁇  ⁇ 。", ("output_invalid_characters",)),
    ("普通的一句话。", "⁇⁇更⁇", ("output_invalid_characters",)),
    ("普通的一句话。", "更" + "?" * 15, ("output_punctuation",)),
    ("普通的一句话。", "？" * 15 + "更。", ("output_punctuation",)),
    ("はいはいと言ってから歌って。", "想" * 15, ("output_repetition",)),
    ("ラ" * 6 + "って歌って。", "啦" * 12, ("output_repetition",)),
    ("ラ" * 12 + "って歌って。", "唱：" + "啦" * 30 + "。", ("output_repetition",)),
])
def test_conspicuous_repetitive_or_invalid_generations_are_rejected(source, target, expected):
    assert output_quality_issues(source, target) == expected
    assert candidate_issues(source, target, "ja_to_zh") == expected


@pytest.mark.parametrize("source,target", [
    ("私の心そっと覗いてみませんか。", "要不要悄悄看看我的内心？"),
    ("考えてみてください。", "想一想吧。"),
    ("ははは！", "哈哈哈！"),
    ("はいはい。", "好，好。"),
    ("好，好。", "はいはい。"),
    ("やった、やった！", "成功了，成功了！"),
    ("絶対にだめ、だめ、だめ。", "绝对不行，不行，不行。"),
    ("私の心そっと覗いてみませんか。", "想" * 11),
    ("私の心そっと覗いてみませんか。", "想" * 12 + "要不要看一下我的心"),
    ("普通的一句话。", "心愿" * 6 + "是什么呢"),
    ("普通的一句话。", "想要看看" * 5),
    ("下一句话很长。", "これはごく普通の日本語の文章です。"),
    ("下一句话很长。", "すもももももももものうち。"),
    ("这是编号。", "111111111111111111111111"),
    ("これは番号です。", "一" * 24),
    ("这是编号。", "１２３" * 12),
    ("哈" * 12, "は" * 20),
    ("は" * 6, "哈" * 12 + "！"),
    ("はい、" * 6, "是的，" * 6),
    ("想要看看" * 6, "覗いてみたい" * 6),
    ("？？？" * 8, "???" * 8),
    ("\ufffd" * 15, "\ufffd" * 15),
    ("普通的话。", "🙂" * 12),
    ("元気ですか？", "你好吗？？？"),
    ("你好吗？？？", "お元気ですか？？？"),
    ("彼は「⁇」と書いた。", "他写了“⁇”这个符号。"),
    ("彼は「⁇⁇⁇」と書いた。", "“⁇⁇⁇”"),
    ("彼は「⁇⁇⁇⁇⁇⁇」と書いた。", "“⁇⁇⁇⁇⁇⁇”"),
    ("⁇⁇⁇", "⁇ ⁇ ⁇"),
    ("なぜ" + "？" * 6, "为什么" + "？" * 12),
    ("普通的一句话。", "你到底在干什么啊" + "？" * 12),
    ("ラ" * 12 + "って歌って。", "唱：" + "啦" * 12 + "。"),
    ("「" + "ラ" * 12 + "」って歌って。", "唱：" + "啦" * 12 + "。"),
    ("「ララララララララララララ」と一緒に歌ってください。", "唱：啦啦啦啦啦啦啦啦啦啦啦啦。"),
    ("願い" * 6 + "って歌って。", "唱：" + "心愿" * 6 + "。"),
])
def test_quality_check_preserves_language_emphasis_numbers_and_source_repetitions(source, target):
    assert output_quality_issues(source, target) == ()


def test_quality_check_catches_a_cycle_split_by_stream_whitespace():
    assert output_quality_issues("私の心そっと覗いてみませんか 欲し。", " 想 " * 15) == (
        "output_repetition",)


def test_garbage_output_is_classified_before_numeric_or_semantic_comparison():
    assert candidate_issues("100円で売れたらいいな。", "?" * 100, "ja_to_zh") == (
        "output_punctuation",)


@pytest.mark.parametrize("source,target,prior", [
    ("そっちより、こっちの時間にやるほうが伸びそう。", "比起下午，上午播可能更好。", ["昼の人数を数えた。", "夕方のほうが多かった。"]),
    ("こっちの時間にしよう。", "选下午吧。", ["昼は空いていた。", "夕方も空いている。"]),
    ("こちらでやろう。", "早上在这里做吧。", []),
    ("そちらにしよう。", "下午做吧。", ["朝なら空いている。"]),
    ("あっちより、こっちがよさそう。", "早上或半夜可能更好。", ["午前なら空いている。"]),
    ("あちらでやろう。", "上午做吧。", ["午前の予定がある。", "別の話です。", "続きは未定です。"]),
])
def test_time_deictic_rejects_only_ungrounded_dayparts(source, target, prior):
    assert context_grounding_issues(source, target, prior) == ("ungrounded_time_reference",)


@pytest.mark.parametrize("source,target,prior", [
    ("そっちより、こっちの時間にやるほうが伸びそう。", "比起那时，这个时段播可能更好。", ["昼と夕方を比較している。"]),
    ("こっちの時間にしよう。", "选下午吧。", ["午前は会議です。", "午後は空いています。"]),
    ("こちらにしよう。", "上午吧。", ["午前なら空いている。"]),
    ("あちらにしよう。", "白天吧。", ["日中なら空いている。"]),
    ("こちらにしよう。", "白天吧。", ["午前なら空いている。"]),
    ("こっちでやろう。", "傍晚在这边做吧。", ["夕方がよさそうです。"]),
    ("こっちでやろう。", "晚上在这边做吧。", ["夕方がよさそうです。"]),
    ("こっちでやろう。", "晚上在这边做吧。", ["夜なら空いている。"]),
    ("そちらでやろう。", "夜里在那边做吧。", ["晩なら空いている。"]),
    ("こっちのほうがよさそう。", "比起白天，傍晚好一些。", ["昼と夕方を比較した。"]),
    ("こっちのほうがよさそう。", "比起傍晚，白天好一些。", ["昼と夕方を比較した。"]),
    ("こちらでやろう。", "就在这边做吧。", []),
    ("午後ならこっちがいい。", "上午在这边吧。", []),
    ("明日はこっちでやろう。", "明天下午在这里做吧。", []),
    ("二十一日はこっちでやろう。", "21日下午在这里做吧。", []),
    ("こっちで9時半に始めよう。", "上午9点半在这里开始吧。", []),
    ("こちらで９：３０に始めよう。", "上午9:30在这里开始吧。", []),
    ("この時間のほうがいい。", "下午更好。", []),
    ("这边比那边好。", "上午比下午好。", []),
    ("コッチさんの話です。", "上午讨论科奇先生的事。", []),
    ("ソッチという曲を聴いた。", "白天听了名叫索奇的歌曲。", []),
    ("こちらの曲は「モーニング」です。", "这首歌叫《早上》。", []),
    ("こっちのアフタヌーンティーがいい。", "这边的下午茶不错。", []),
])
def test_time_grounding_preserves_basis_ambiguity_and_out_of_scope_inputs(source, target, prior):
    # Accepting means this deliberately narrow gate abstained, not that the
    # candidate's time, referent, or overall translation was proved correct.
    assert context_grounding_issues(source, target, prior) == ()
