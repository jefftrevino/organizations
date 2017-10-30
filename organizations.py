from abjad import *
from organizationsStylesheet import *
from random import sample, seed, randint
from itertools import combinations
from fractions import gcd

counts = [3, 2, 1, 0]


def lcm_of_two(a, b):
    return (a * b) // gcd(a, b)


def lcm_of_list(numbers):
    if len(numbers) == 2:
        out = lcm_of_two(numbers[0], numbers[1])
        return out
    elif len(numbers) > 2:
        input_list = [lcm_of_two(numbers[0], numbers[1])] + numbers[2:]
        from_beneath = lcm_of_list(input_list)
        return from_beneath


def make_bar_divs_from_full_dur(full_dur):
    bars = full_dur // Duration(4, 4)
    remainder = full_dur - Duration(bars, 1)
    if remainder.denominator == 2:
        remainder = Duration(remainder).with_denominator(4)
    return bars, remainder


def listMultiplesBeforeLCM(x, lcm):
    return list(range(0, lcm+1, x))


def getPairwiseDiffs(theList):
    diffs = []
    for x in range(len(theList)-1):
        diffs.append(theList[x+1] - theList[x])
    return diffs


def count_series_from_durations(divisors, lcm):
    multiple_lists = []
    for divisor in divisors:
        multiple_lists.append(listMultiplesBeforeLCM(divisor, lcm))
    interlaced = sequencetools.interlace_sequences(multiple_lists)
    interlaced.sort()
    interlaced = sequencetools.remove_repeated_elements(interlaced)
    count_series = getPairwiseDiffs(interlaced)
    return count_series


def make_talea(divisors, denom, lcm):
    count_series = count_series_from_durations(divisors, lcm)
    ptalea = rhythmmakertools.Talea(counts=count_series, denominator=denom)
    return ptalea


def make_divisions(phrase_length):
    bars, remainder = make_bar_divs_from_full_dur(phrase_length)
    divisions = [mathtools.NonreducedFraction(4, 4)] * bars
    if remainder:
        divisions.append(remainder)
    return divisions


def create_ratio_markup(divisors):
    theString = str(divisors[0])
    for d in divisors[1:]:
        theString += " against " + str(d)
    return Markup(theString, direction=Up)


def create_stream_rhythm(divisor, phrase_length, denom):
    division_numerators = [divisor] * (phrase_length//Duration(divisor, denom))
    division_numerators *= 2
    division_numerators.append(16)
    orig_divisions = make_divisions(phrase_length)
    divisions = orig_divisions * 2
    divisions.append(NonreducedFraction(4, 4))
    talea = rhythmmakertools.Talea(counts=division_numerators, denominator=denom)
    maker = rhythmmakertools.TaleaRhythmMaker(talea)
    selections = maker(divisions)
    voice = Voice(selections)
    maker = LeafMaker()
    pitches = len(orig_divisions) * [None]
    rests = maker(pitches, orig_divisions)
    voice.extend(rests)
    divisions.extend(orig_divisions)
    result = mutate(voice[:]).split(divisions)
    for s, d in zip(result, divisions):
        mutate(s).wrap(Measure(d, []))
    return voice


def create_interleaved_rhythm(divisors, lcm, denominator, phrase_length):
    ptalea = make_talea(divisors, denominator, lcm)
    maker = rhythmmakertools.TaleaRhythmMaker(
        ptalea,
        tuplet_spelling_specifier=rhythmmakertools.TupletSpellingSpecifier(simplify_redundant_tuplets=True)
        )
    divisions = make_divisions(phrase_length)
    selections = maker(divisions)
    voice = Voice(selections)
    return voice


def choose_factors(length, num_divisors):
    divisors = mathtools.divisors(length)
    divisors = divisors[2:-1]
    divisors = divisors[-3:]
    return divisors


def pitch_tie_chain(chain, pitch):
    if not isinstance(chain[0], Rest):
        for x in chain:
            x.written_pitch = pitch


def pitch_selections(selections, pitch_list, starting_index, reverse=False):
    cycle = datastructuretools.CyclicTuple(pitch_list)
    for x, tie in enumerate(iterate(selections).by_logical_tie()):
        index = x + starting_index
        pitch_tie_chain(tie, cycle[index])


def add_text_spanner_to_leaves(text_string, leaves):
    text_spanner = spannertools.TextSpanner()
    markup = Markup(text_string).italic().bold()
    override(text_spanner).text_spanner.bound_details__left__text = markup
    override(text_spanner).text_spanner.bound_details__left__stencil_align_dir_y = 0
    attach(text_spanner, leaves[:])


def deoctavize_pitches(pitches):
    analysis_pitches = []
    for p in pitches:
        if p >= 24 and p < 36:
            analysis_pitches.append(p - 24)
        elif p >= 12 and p < 24:
            analysis_pitches.append(p - 12)
        elif p < 0:
            analysis_pitches.append(p + 12)
        else:
            analysis_pitches.append(p)
    return analysis_pitches


def same_pitch_class(p1, p2):
    if p1.pitch_class == p2.pitch_class:
        return True
    return False


def is_second(p1, p2):
    if abs(p1.number - p2.number) <= 2:
        return True
    return False


def acceptable(pitches):
    for combo in combinations(pitches, 2):
        p1 = combo[0]
        p2 = combo[1]
        interval_num = abs(p1.number - p2.number)
        if interval_num < 7 or interval_num == 6 or same_pitch_class(p1, p2) or is_second(p1, p2):
            return False
    return True


def choose_starting_pitch_indexes(pitch_lists):
    # returns index of initial pitches
    pitches = []
    for plist in pitch_lists:
        pitches.append(sample(plist, 1)[0])
    while not acceptable(pitches):
        pitches = []
        for plist in pitch_lists:
            pitches.append(sample(plist, 1)[0])
    indexes = []
    for pair in zip(pitch_lists, pitches):
        plist = pair[0]
        pitch = pair[1]
        indexes.append(plist.index(pitch))
    return indexes


def add_stream_voices_to_group(phrase_length, divisors, denominator, staff_group, scale_index):
    scale = scales[scale_index]
    pitch_lists = make_pitch_lists(scale)
    stream_rhythm_voices = []
    starting_pitch_indexes = choose_starting_pitch_indexes(pitch_lists)
    for x in range(3):
        starting_pitch_index = starting_pitch_indexes[x]
        stream_rhythm_voices.append(create_stream_rhythm(divisors[x], phrase_length, denominator))
        attach(scale.key_signature, stream_rhythm_voices[x][0])
    for x,v in enumerate(stream_rhythm_voices):
        pitches = pitch_lists[x]
        index_offset = randint(0, len(pitches) - 1)
        pitch_selections(v, pitches, starting_pitch_indexes[x], reverse=False)
    top_staff = staff_group[0]
    bottom_staff = staff_group[1]
    top_staff[0].extend(stream_rhythm_voices[0][:])
    top_staff[1].extend(stream_rhythm_voices[1][:])
    bottom_staff[0].extend(stream_rhythm_voices[2][:])


def add_phrase_to_group(phrase_length, denominator, staff_group, x):
    divisors = choose_factors(phrase_length, 3)
    lcm = lcm_of_list(divisors)
    divisors.sort()
    phrase_length = Duration(mathtools.NonreducedFraction(phrase_length, denominator))
    add_stream_voices_to_group(phrase_length, divisors, denominator, staff_group, x)


def format_voices(staff_group):
    top_staff = staff_group[0]
    bottom_staff = staff_group[1]
    voices = [v for v in iterate(top_staff).by_class(Voice)]
    voice_one_leaves = iterate(voices[0]).by_leaf()
    for note_run in iterate(voice_one_leaves).by_run(prototype=Note):
        attach(indicatortools.LilyPondCommand('voiceOne'), note_run[0])
    voice_one_leaves = iterate(voices[0]).by_leaf()
    for rest_run in iterate(voice_one_leaves).by_run(prototype=Rest):
        for rest in rest_run:
            mutate([rest]).replace([Skip(rest.written_duration)])
    voice_one_leaves = iterate(voices[0]).by_leaf()
    for skip_run in iterate(voice_one_leaves).by_run(prototype=Skip):
        attach(indicatortools.LilyPondCommand('oneVoice'), skip_run[0])
    voice_two_leaves = iterate(voices[1]).by_leaf()
    for note_run in iterate(voice_two_leaves).by_run(prototype=Note):
        attach(indicatortools.LilyPondCommand('voiceTwo'), note_run[0])
    voice_two_leaves = iterate(voices[1]).by_leaf()
    for rest_run in iterate(voice_two_leaves).by_run(prototype=Rest):
        attach(indicatortools.LilyPondCommand('oneVoice'), rest_run[0]) 
    top_dynamic = Dynamic('ppp')
    attach(top_dynamic, top_staff[0][0][0])
    bottom_dynamic = Dynamic('ppp')
    attach(bottom_dynamic, bottom_staff[0][0][0])
    top_staff[1]


def make_pitch_lists(scale):
    pitch_lists = [list(scale.create_named_pitch_set_in_pitch_range(x)) for x in pitch_ranges]
    pitch_lists = [sorted(x) for x in pitch_lists]
    for l in pitch_lists:
        l.reverse()
    return pitch_lists

def make_empty_score():
    score = Score([])
    staff_group = StaffGroup(context_name='PianoStaff')
    score.append(staff_group)
    staff_group.extend([Staff(), Staff()])
    staff_group[0].is_simultaneous = True
    staff_group[0].extend([Voice(), Voice()])
    staff_group[1].append(Voice())
    attach(Clef('bass'), staff_group[1])
    return score

def make_score(phrase_lengths, denominator):
    score = make_empty_score()
    staff_group = score[0]
    for x, phrase_length in enumerate(phrase_lengths):
        add_phrase_to_group(phrase_length, denominator, staff_group, x)
    format_voices(staff_group)
    return score

    return phrase_lengths

def get_pitch_class_string(abbreviation):
    base = abbreviation[0]
    rest = abbreviation[1:]
    if rest == 'f':
        rest = "b"
    elif rest == 's':
        rest = '#'
    base = base.upper()
    return markuptools.MarkupCommand("left-align", "\\teeny", base+rest)

def isTiedTo(note):
    theTie = inspect(note).get_logical_tie()
    if 1 == len(theTie):
        return False
    elif 1 < len(theTie) and theTie.head == note:
        return False
    return True


def add_markup_to_illegible_note(note):
    if not isinstance(note, Note):
        return None
    padding = 0.5
    if note.written_pitch.number >= 31 and not isTiedTo(note):
        class_abbreviation = str(note.written_pitch.pitch_class)
        letter = get_pitch_class_string(class_abbreviation)
        padded_markup = markuptools.MarkupCommand('pad-markup', schemetools.Scheme( padding ), letter)
        markup = markuptools.Markup(padded_markup, direction=Up)
        attach(markup, note)
    elif note.written_pitch.number <= -27 and not isTiedTo(note):
        class_abbreviation = str(note.written_pitch.pitch_class)
        letter = get_pitch_class_string(class_abbreviation)
        padded_markup = markuptools.MarkupCommand('pad-markup', schemetools.Scheme( padding ), letter)
        markup = markuptools.Markup(padded_markup, direction=Down)
        attach(markup, note)

def format_score(score):
    doublebar = indicatortools.BarLine('|.')
    for voice in iterate(score).by_class(Voice):
        attach(indicatortools.LilyPondCommand('compressFullBarRests'), voice)
    attach(doublebar, score[0][0][0][-1][-1])

def are_in_same_tie(leaf1, leaf2):
    the_tie = inspect(leaf1).get_logical_tie()
    if leaf2 in the_tie:
        return True
    return False

def get_tie_leaves_in_measure(tie_chain, measure):
    leaves = []
    for leaf in tie_chain:
        parentage = inspect(leaf).get_parentage()
        if parentage[1] == measure:
            leaves.append(leaf)
    return leaves


def fuse_downbeat_dotted_halves(measure):
    if sum([x.written_duration for x in measure[:2]]) == Duration(3,4) and are_in_same_tie(measure[0], measure[1]):
        mutate(measure[:2]).fuse()

def fuse_upbeat_dotted_halves(measure):
    if sum([x.written_duration for x in measure[-2:]]) == Duration(3,4) and are_in_same_tie(measure[-1], measure[-2]):
        mutate(measure[-2:]).fuse()

def divide_sixteenths_bar(measure):
    m = metertools.Meter(measure.time_signature)
    mutate(measure).rewrite_meter(m)
    # quarters = measure.target_duration // Duration(1,4)
    # remainder = measure.target_duration - (quarters * Duration(1,4))
    # divisions = [mathtools.NonreducedFraction(1,4)] * quarters
    # if remainder:
    #     divisions.append(remainder)
    # mutate(measure[:]).split(divisions)


def impose_meter(score):
    four = metertools.Meter('(4/4 ((2/4 (1/4 1/4)) (2/4 (1/4 1/4))))')
    three = metertools.Meter(Duration(3, 4))
    eleven = metertools.Meter("(11/16 (1/4 1/4 3/16))")
    fifteen = metertools.Meter('(15/16 (1/4 1/4 1/4 3/16))')
    meters = {
        TimeSignature(mathtools.NonreducedFraction(4, 4)): four,
        TimeSignature(mathtools.NonreducedFraction(3, 4)): three,
        TimeSignature(mathtools.NonreducedFraction(11, 16)): eleven,
        TimeSignature(mathtools.NonreducedFraction(15, 16)): fifteen
    }
    for x, measure in enumerate(iterate(score).by_class(Measure)):
        if len(measure) == 1 and isinstance(measure[0], Rest):
            continue
        elif len(measure) >= 3 and measure.time_signature not in meters.keys():
            fuse_downbeat_dotted_halves(measure)
            fuse_upbeat_dotted_halves(measure)
        else:
            sig = measure.time_signature
            if sig in meters.keys():
                mutate(measure).rewrite_meter(meters[sig], maximum_dot_count = 1)

def build_phrase_length_dict(shortest, longest):
    phrase_dict = {}
    for x in range(shortest,longest+1):
        factors = choose_factors(x,3)
        if len(factors) == 3:
            phrase_dict[x] = factors
    return phrase_dict

def get_next_length_candidates_from_factors(phrase_dict, factors):
    candidate_lengths = []
    for factor in factors:
        candidate_lengths.extend([x for x in phrase_dict if factor in phrase_dict[x]])
    candidate_lengths = set(candidate_lengths)
    return candidate_lengths

def eliminate_repetitions_from_candidates(last_phrase_length, phrase_lengths, candidate_lengths):
    originals = candidate_lengths
    to_not_choose = set()
    to_not_choose.update(phrase_lengths)
    to_not_choose.update([last_phrase_length])
    pruned = candidate_lengths.difference(to_not_choose)
    if len(pruned) <= 2:
        return originals
    return pruned


def choose_phrase_lengths(shortest, longest, piece_dur_in_denom_units):
    # dovetail selection by common factor between phrase lengths, i.e.
    # 12 [3,4,6], 24 [6,8,12] (via 6 as voice length), 48 [12, 16, 24] (via 12 as voice length), etc.
    # length_seed = 0
    phrase_dict = build_phrase_length_dict(shortest, longest)
    key_list = sorted(list(phrase_dict.keys()))
    # key_list.reverse()
    phrase_lengths = []
    for key in key_list:
        if sum(phrase_lengths) < piece_dur_in_denom_units/2.0:
            phrase_lengths.append(key)
        else:
            break
    return phrase_lengths

def mark_illegible_notes(score):
    for tie in iterate(score[0][0][0]).by_logical_tie():
            add_markup_to_illegible_note(tie[0])

def render_notation(shortest_phrase, longest_phrase, target_piece_dur_in_s):
    # arguments:
    # shortest_phrase: shortest phrase in denom units
    # longest_phrase: longest phrase in denom units
    # target_piece_dur_in_s: target piece dur in seconds
    bpm = 80.0
    tempo_denominator = 4
    piece_dur_in_beats = (target_piece_dur_in_s / 60.0) * bpm
    rhythm_denominator = 16
    piece_dur_in_denom_units = piece_dur_in_beats * (rhythm_denominator / 4)
    phrase_lengths = choose_phrase_lengths(shortest_phrase, longest_phrase, piece_dur_in_denom_units)
    # for each phrase length, produce three voices with a base division of a thirty-second note.
    score = make_score(phrase_lengths, rhythm_denominator)
    impose_meter(score)
    tempo = MetronomeMark((1,tempo_denominator), bpm)
    attach(tempo, score[0][0][0])
    format_score(score)
    mark_illegible_notes(score)
    # print some stats about the notation generated
    score_duration_in_bars = float(inspect(score[0][0][0]).get_duration())
    score_duration_in_beats = score_duration_in_bars * tempo_denominator
    score_duration_in_minutes= float(score_duration_in_beats) / bpm
    print "The duration of the composition is", score_duration_in_beats, "beats.", "\n"
    print "At a tempo of", bpm, "bpm, this is ", score_duration_in_minutes, "minutes long."
    return score


pitch_ranges = [
    pitchtools.PitchRange('[D5, A6]'),
    pitchtools.PitchRange('[F4, C5]'),
    pitchtools.PitchRange('[E3, B3]'),
    ]

scales = datastructuretools.CyclicTuple([
    tonalanalysistools.Scale(('b', 'major')),
    tonalanalysistools.Scale(('b', 'minor')),
    tonalanalysistools.Scale(('g', 'major')),
    tonalanalysistools.Scale(('g', 'minor')),
    tonalanalysistools.Scale(('ef', 'major')),
    tonalanalysistools.Scale(('ef', 'minor')),
    ])

seed(10)
score = render_notation(50, 130, 300)
sketch = make_lilypond_file(score)
show(sketch)
play(sketch)