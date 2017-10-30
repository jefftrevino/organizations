from abjad import *
from os import path, getcwd


def make_lilypond_file(component):
    for voice in iterate(component).by_class(Voice):
        voice.remove_commands.append('Forbid_line_break_engraver')
    # override(component).spacing_spanner.strict_grace_spacing = True
    # override(component).spacing_spanner.strict_note_spacing = True
    # override(component).spacing_spanner.uniform_stretching = True
    override(component).stem.length = 8.25
    override(component).text_script.outside_staff_padding = 1
    override(component).time_signature.stencil = False
    override(component).tuplet_bracket.bracket_visibility = True
    override(component).tuplet_bracket.minimum_length = 3
    override(component).tuplet_bracket.outside_staff_padding = 1.5
    override(component).tuplet_bracket.padding = 1.5
    override(component).tuplet_bracket.springs_and_rods = \
        schemetools.Scheme('ly:spanner::set-spacing-rods', verbatim=True)
    override(component).tuplet_bracket.staff_padding = 2.25
    override(component).tuplet_number.text = \
        schemetools.Scheme('tuplet-number::calc-fraction-text', verbatim=True)
    setting(component).proportional_notation_duration = SchemeMoment((1, 16))
    setting(component).tuplet_full_length = True
    directory = path.abspath( getcwd() )
    fontTree = directory+'/fontTree.ly'
    lilypond_file = lilypondfiletools.LilyPondFile.new(component, includes=[fontTree])
    lilypond_file.layout_block.indent = 0
    return lilypond_file