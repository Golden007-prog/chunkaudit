"""Build the sample dataset in ``chunkaudit/data/sample``.

The five short articles below were written for this project, so they can be shipped with it.
Each question lists the exact text that answers it. The script checks that every piece of
evidence really occurs in its article before writing the files.

    python scripts/build_sample.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "chunkaudit" / "data" / "sample"

DOCUMENTS = {
    "tides": """
Tides are the regular rise and fall of sea level along a coast. They are caused mainly by the gravitational pull of the Moon, with a smaller contribution from the Sun. The side of the Earth facing the Moon is pulled slightly more strongly than the centre of the planet, and the far side slightly less, so the oceans bulge on both sides.

As the Earth rotates, a coastline passes through both bulges. Most places therefore see two high tides and two low tides in a little under 25 hours. The extra time beyond a full day exists because the Moon moves along its orbit while the Earth turns, so a point on the surface needs about 50 more minutes to catch up with it.

The height of the tide changes through the month. When the Sun, Earth and Moon are lined up, at new moon and full moon, their pulls add together and produce spring tides, which have the largest range. When the Moon is at a right angle to the Sun, at the first and last quarter, the pulls partly cancel and produce neap tides, which have the smallest range.

Local geography matters as much as astronomy. A wide, shallow bay that narrows inland can funnel the incoming water and amplify the tide many times over. The Bay of Fundy in Canada is the best known example, with a range that can exceed 15 metres. In contrast, enclosed seas such as the Mediterranean have tides of only a few tens of centimetres.
""",
    "sourdough": """
Sourdough bread is raised by a starter instead of commercial yeast. A starter is a mixture of flour and water that hosts a stable community of wild yeasts and lactic acid bacteria. The yeasts produce carbon dioxide, which makes the dough rise, while the bacteria produce lactic and acetic acid, which give the bread its sour taste.

A starter has to be fed to stay active. Feeding means discarding part of the mixture and adding fresh flour and water, usually once a day at room temperature. A starter kept in the refrigerator slows down and can be fed about once a week. It is ready to use when it has roughly doubled in volume a few hours after a feed and is full of bubbles.

Temperature controls both speed and flavour. A warm dough, around 26 to 28 degrees Celsius, ferments quickly and tastes milder. A cool dough ferments slowly, and a long rest in the refrigerator, often overnight, gives a more pronounced sour flavour because acid keeps building while the yeast activity drops.

Baking a loaf with a crisp crust depends on steam during the first part of the bake. Many home bakers use a heavy lidded pot, which traps the moisture released by the dough itself. The lid is removed for the last part of the bake so the crust can dry and brown. The acids in the bread also slow down staling and mould, so a sourdough loaf keeps longer than most yeasted bread.
""",
    "bicycle-gears": """
A bicycle with derailleur gears lets the rider change how far the bike travels for each turn of the pedals. The gear ratio is the number of teeth on the front chainring divided by the number of teeth on the rear sprocket. A large chainring with a small sprocket gives a high gear, which covers more distance per pedal turn but needs more force.

Low gears are for climbing. With a small chainring and a large sprocket the rider spins the pedals quickly while the bike moves slowly, which keeps the effort manageable on a steep hill. Shifting to a lower gear before the slope begins is easier on the drivetrain than shifting under heavy load halfway up.

Most riders are efficient at a cadence of roughly 80 to 100 pedal revolutions per minute. The purpose of changing gear is to keep the cadence in that range as the terrain and wind change, not to push as hard as possible. Pedalling slowly in a very high gear strains the knees and tires the legs sooner.

Some combinations should be avoided. Using the largest chainring with the largest sprocket, or the smallest with the smallest, runs the chain at a sharp angle. This is called cross-chaining, and it increases wear and noise. A chain also stretches with use, and replacing it in time protects the more expensive sprockets and chainrings from wearing out early.
""",
    "honeybees": """
A honeybee colony is a single family built around one queen. In summer a healthy colony holds tens of thousands of worker bees, a few hundred drones and the queen. The workers are all female and do every job in the hive except laying eggs, while the drones are males whose only role is to mate with new queens.

A worker's job changes with age. For the first weeks of her life she works inside the hive, cleaning cells, feeding larvae, building comb and guarding the entrance. Only in the last part of her life does she become a forager and fly out to collect nectar, pollen and water. A summer worker lives for about six weeks, while workers raised in autumn can survive the whole winter.

Foragers share the location of good flowers with a movement called the waggle dance. The angle of the dance relative to vertical on the comb shows the direction of the food relative to the Sun, and the length of the waggle run shows the distance. Other workers follow the dancer closely and then fly to the same patch.

Colonies reproduce by swarming. In late spring the old queen leaves with roughly half of the workers and settles in a temporary cluster, often on a tree branch, while scout bees search for a new home. The bees that stay behind raise a new queen from young larvae fed on royal jelly.
""",
    "lighthouses": """
A lighthouse marks a dangerous stretch of coast or the entrance to a harbour. Each one shows a distinct pattern of flashes, called its characteristic, so that a navigator can tell it apart from its neighbours. Nautical charts print the pattern, the colour of the light and the period in seconds next to each lighthouse.

The distance at which a light can be seen depends on its height as well as its brightness. Because the Earth is curved, a light placed higher above the sea stays above the horizon for a ship that is further away. This is why lighthouses on low coasts are tall towers, while those on high cliffs can be short buildings.

The Fresnel lens transformed lighthouses in the 1820s. Instead of one thick piece of glass, it uses concentric rings of prisms that bend the light from a single lamp into a narrow horizontal beam. The design captures far more of the lamp's light than the mirrors used before it and weighs much less than a solid lens of the same power.

For most of their history lighthouses needed keepers, who trimmed the wicks, wound the clockwork that turned the lens and kept the glass clean. Electric lamps and automatic controls removed the need for a resident crew during the twentieth century. Many lighthouses still operate today, although satellite navigation has made ships far less dependent on them.
""",
}

# (document id, question, evidence)
QUESTIONS = [
    (
        "tides",
        "What mainly causes tides?",
        "They are caused mainly by the gravitational pull of the Moon, with a smaller contribution from the Sun.",
    ),
    (
        "tides",
        "Why does a full cycle of tides take longer than 24 hours?",
        "The extra time beyond a full day exists because the Moon moves along its orbit while the Earth turns, so a point on the surface needs about 50 more minutes to catch up with it.",
    ),
    (
        "tides",
        "When do neap tides happen?",
        "When the Moon is at a right angle to the Sun, at the first and last quarter, the pulls partly cancel and produce neap tides, which have the smallest range.",
    ),
    (
        "tides",
        "How large can the tidal range be in the Bay of Fundy?",
        "The Bay of Fundy in Canada is the best known example, with a range that can exceed 15 metres.",
    ),
    (
        "sourdough",
        "What gives sourdough bread its sour taste?",
        "The yeasts produce carbon dioxide, which makes the dough rise, while the bacteria produce lactic and acetic acid, which give the bread its sour taste.",
    ),
    (
        "sourdough",
        "How often does a starter kept in the refrigerator need feeding?",
        "A starter kept in the refrigerator slows down and can be fed about once a week.",
    ),
    (
        "sourdough",
        "Why does an overnight rest in the refrigerator make the bread more sour?",
        "A cool dough ferments slowly, and a long rest in the refrigerator, often overnight, gives a more pronounced sour flavour because acid keeps building while the yeast activity drops.",
    ),
    (
        "sourdough",
        "Why do home bakers bake bread in a lidded pot?",
        "Many home bakers use a heavy lidded pot, which traps the moisture released by the dough itself.",
    ),
    (
        "bicycle-gears",
        "How is the gear ratio of a bicycle calculated?",
        "The gear ratio is the number of teeth on the front chainring divided by the number of teeth on the rear sprocket.",
    ),
    (
        "bicycle-gears",
        "When should a rider shift down for a hill?",
        "Shifting to a lower gear before the slope begins is easier on the drivetrain than shifting under heavy load halfway up.",
    ),
    (
        "bicycle-gears",
        "What pedalling cadence is efficient for most riders?",
        "Most riders are efficient at a cadence of roughly 80 to 100 pedal revolutions per minute.",
    ),
    (
        "bicycle-gears",
        "What is cross-chaining and why is it a problem?",
        "Using the largest chainring with the largest sprocket, or the smallest with the smallest, runs the chain at a sharp angle. This is called cross-chaining, and it increases wear and noise.",
    ),
    (
        "honeybees",
        "What is the role of drones in a honeybee colony?",
        "The workers are all female and do every job in the hive except laying eggs, while the drones are males whose only role is to mate with new queens.",
    ),
    (
        "honeybees",
        "How long does a worker bee live in summer?",
        "A summer worker lives for about six weeks, while workers raised in autumn can survive the whole winter.",
    ),
    (
        "honeybees",
        "How does the waggle dance tell other bees where the food is?",
        "The angle of the dance relative to vertical on the comb shows the direction of the food relative to the Sun, and the length of the waggle run shows the distance.",
    ),
    (
        "honeybees",
        "What does the old queen do when a colony swarms?",
        "In late spring the old queen leaves with roughly half of the workers and settles in a temporary cluster, often on a tree branch, while scout bees search for a new home.",
    ),
    (
        "lighthouses",
        "How can a navigator tell one lighthouse from another?",
        "Each one shows a distinct pattern of flashes, called its characteristic, so that a navigator can tell it apart from its neighbours.",
    ),
    (
        "lighthouses",
        "Why are lighthouses on low coasts built as tall towers?",
        "Because the Earth is curved, a light placed higher above the sea stays above the horizon for a ship that is further away. This is why lighthouses on low coasts are tall towers, while those on high cliffs can be short buildings.",
    ),
    (
        "lighthouses",
        "How does a Fresnel lens produce a narrow beam?",
        "Instead of one thick piece of glass, it uses concentric rings of prisms that bend the light from a single lamp into a narrow horizontal beam.",
    ),
    (
        "lighthouses",
        "What work did lighthouse keepers do?",
        "For most of their history lighthouses needed keepers, who trimmed the wicks, wound the clockwork that turned the lens and kept the glass clean.",
    ),
]


def main() -> None:
    documents = {doc_id: text.strip() for doc_id, text in DOCUMENTS.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for doc_id, text in documents.items():
            handle.write(json.dumps({"id": doc_id, "text": text}) + "\n")
    with (OUT / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for number, (doc_id, question, evidence) in enumerate(QUESTIONS, start=1):
            if documents[doc_id].count(evidence) != 1:
                raise SystemExit(f"evidence of question {number} must occur exactly once")
            row = {
                "id": f"q{number:02d}",
                "question": question,
                "doc_id": doc_id,
                "evidence": evidence,
            }
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {len(documents)} documents and {len(QUESTIONS)} questions to {OUT}")


if __name__ == "__main__":
    main()
