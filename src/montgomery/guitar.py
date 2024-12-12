from typing import List, Optional

from montgomery.helper import print_verbose


class Pitch:
    NOTE_NAME_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(self, note_name: str, octave: int):
        self.note_name = note_name
        self.octave = octave

    def __init__(self, as_str: str):
        if len(as_str) < 2 or len(as_str) > 3:
            raise ValueError(f"Invalid pitch string: {as_str}")

        as_str = as_str.replace("\u266F", "#")  # "♯" symbol
        if as_str[:-1].upper() not in Pitch.NOTE_NAME_ORDER:
            raise ValueError(f"Invalid note name: {as_str[:-1]}")
        self.note_name = as_str[:-1].upper()
        if not as_str[-1].isdigit() or int(as_str[-1]) < 0:
            raise ValueError(f"Invalid octave: {as_str[-1]}")
        self.octave = int(as_str[-1])

    def __repr__(self):
        return f"{self.note_name}{self.octave}"

    def to_int(self) -> int:
        return Pitch.NOTE_NAME_ORDER.index(self.note_name) + 12 * (self.octave)

    def __eq__(self, other: "Pitch") -> bool:
        if not isinstance(other, Pitch):
            return False
        return self.to_int() == other.to_int()

    def __lt__(self, other: "Pitch") -> bool:
        return self.to_int() < other.to_int()

    def __le__(self, other: "Pitch") -> bool:
        return self.to_int() <= other.to_int()

    def __gt__(self, other: "Pitch") -> bool:
        return self.to_int() > other.to_int()

    def __ge__(self, other: "Pitch") -> bool:
        return self.to_int() >= other.to_int()

    def subtract(self, other: "Pitch") -> int:
        return self.to_int() - other.to_int()


class GuitarTab:
    MAX_FRET_INDEX = 24
    BASE_STRINGS = [
        Pitch("E2"),
        Pitch("A2"),
        Pitch("D3"),
        Pitch("G3"),
        Pitch("B3"),
        Pitch("E4"),
    ]

    def __init__(self, string_index: int, fret_index: int):
        self.string_index = string_index
        self.fret_index = fret_index

    def __repr__(self):
        return f"{GuitarTab.BASE_STRINGS[self.string_index]}: {self.fret_index}"

    def __eq__(self, other: "GuitarTab") -> bool:
        if not isinstance(other, GuitarTab):
            return False
        return (
            self.string_index == other.string_index
            and self.fret_index == other.fret_index
        )

    def __hash__(self) -> int:
        return hash((self.string_index, self.fret_index))

    @staticmethod
    def possible_tabs(pitch: Pitch) -> List["GuitarTab"]:
        possible = []
        for idx, base in enumerate(GuitarTab.BASE_STRINGS):
            diff = pitch.subtract(base)
            if 0 < diff and diff <= GuitarTab.MAX_FRET_INDEX:
                possible.append(GuitarTab(idx, diff))
        return possible


def tabs2string(tabs: List[Optional[GuitarTab]]):
    positions_per_string = [[f"{s.note_name} "] for s in GuitarTab.BASE_STRINGS]
    positions_per_string[-1][0] = positions_per_string[-1][0].lower()
    for tab in tabs:
        for string_idx in range(len(GuitarTab.BASE_STRINGS)):
            if tab == None:
                positions_per_string[string_idx].append("??")
            elif string_idx == tab.string_index:
                fret_index = str(tab.fret_index)
                if (len(fret_index)) == 1:
                    fret_index = f"-{fret_index}"
                positions_per_string[string_idx].append(fret_index)
            else:
                positions_per_string[string_idx].append("--")

    return "\n".join(
        ["--".join(positions) for positions in reversed(positions_per_string)]
    )


class Guitar:
    def __init__(self, fret_positions: List[int]):
        self.fret_positions = Guitar.initialize_fret_position(fret_positions)

    @staticmethod
    def initialize_fret_position(fret_positions: List[int]) -> List[int]:
        fret_positions = list(reversed(sorted(fret_positions)))  # right is 0th fret
        if len(fret_positions) < 2:
            return fret_positions

        print_verbose(
            f"Guitar adjusting fret positions: fret_positions={fret_positions}"
        )
        adjusted_positions = [fret_positions[0], fret_positions[1]]
        for idx in range(2, len(fret_positions)):
            interval = adjusted_positions[-2] - adjusted_positions[-1]
            next_interval = adjusted_positions[-1] - fret_positions[idx]
            if next_interval < (2 / 3) * interval:
                print_verbose(
                    f"removing element: idx={idx}, value={fret_positions[idx]}"
                )
                continue

            elif next_interval > (4 / 3) * interval:
                new_element = adjusted_positions[-1] - interval / 2
                adjusted_positions.append(new_element)
                print_verbose(f"adding extra element: idx={idx}, value={new_element}")

            adjusted_positions.append(fret_positions[idx])

        return adjusted_positions

    def __repr__(self):
        return f"Guitar(fret_positions={self.fret_positions})"

    def get_fret_index(self, x_coordinate: int):
        for i in range(len(self.fret_positions)):
            fret = self.fret_positions[i]
            if i > 0:
                if x_coordinate > fret:
                    return i

        return None


def test_tabs2string():
    tabs = [GuitarTab(2, 12), GuitarTab(4, 15), GuitarTab(3, 14), GuitarTab(3, 12)]
    print(tabs2string(tabs))


if __name__ == "__main__":
    test_tabs2string()
    print(GuitarTab.possible_tabs(Pitch("E3")))
