import cv2

from match import prepare_image, prepare_template, find_template
from collections import namedtuple
from os import path

Icon = namedtuple('Icon', ['image_path', 'description'])

THIS_FOLDER = path.dirname(__file__)
ICONS_DIR = path.join(THIS_FOLDER, 'static', 'eco_icons')
DETECTION_THRESHOLD = 0.35

ICONS = {
    'Green Circle': Icon(image_path=path.join(ICONS_DIR, 'green_circle.jpg'),
                         description='''The Green Dot does not necessarily mean that the packaging is recyclable, 
                         will be recycled or has been recycled. It is a symbol used on packaging in some European 
                         countries and signifies that the producer has made a financial contribution towards 
                         the recovery and recycling of packaging in Europe.'''),
    'Tidyman': Icon(image_path=path.join(ICONS_DIR, 'tidyman.jpg'),
                    description='''This symbol from Keep Britain Tidy asks you not to litter. It doesn't relate to 
                    recycling but is a reminder to be a good citizen, 
                    disposing of the item in the most appropriate manner.'''),
    'Waste Electricals': Icon(image_path=path.join(ICONS_DIR, 'waste_electricals.jpg'),
                              description='''This symbol explains that you should not place the electrical item in the 
                              general waste. Electrical items can be recycled through a number of channels.'''),
    'Paper Card Wood': Icon(image_path=path.join(ICONS_DIR, 'paper_card_wood.jpg'),
                            description='''The Forest Stewardship Council (FSC) logo identifies wood-based products 
                            from well managed forests independently certified in accordance with the rules of the FSC.
                            '''),
    'Mobius Loop': Icon(image_path=path.join(ICONS_DIR,'mobius_loop.jpg'),
                        description='''This indicates that an object is capable of being recycled, not that the object 
                        has been recycled or will be accepted in all recycling collection systems. 
              Sometimes this symbol is used with a percentage figure in the middle to explain that the packaging 
              contains x% of recycled material.'''),
    'PET template': Icon(image_path=path.join(ICONS_DIR,'pet_template.jpg'),
                         description='''1 – PETE – Polyethylene Terephthalate
            The easiest of plastics to recycle. Often used for soda bottles, water bottles and many common food 
            packages. Is recycled into bottles and polyester fibers
            2 – HDPE – High density Polyethylene
            Also readily recyclable – Mostly used for packaging detergents, bleach, milk containers, 
            hair care products and motor oil. Is recycled into more bottles or bags.
            3 – PVC – Polyvinyl Chloride
            This stuff is everywhere – pipes, toys, furniture, packaging – you name it. Difficult to recycle and 
            PVC is a major environmental and health threat.
            4 – LDPE Low-density Polyethylene
            Used for many different kinds of wrapping, grocery bags and sandwich bags and can be recycled into more of 
            the same.
            5 – PP – Polypropylene
            Clothing, bottles, tubs and ropes. Can be recycled into fibers.
            6 – PS – Polystyrene
            Cups, foam food trays, packing peanuts. Polystyrene (also known as styrofoam) is a real problem as it’s 
            bulky yet very lightweight and that makes it difficult to recycle. For example, a carload of expanded 
            polystyrene would weigh next to nothing so there’s not a lot of materials to reclaim, particularly when you 
            take into account the transport getting it to the point of recycling. It can however be reused. 
            Learn more about recycling polystyrene.
            7 – Other
            Could be a mixture of any and all of the above or plastics not readily recyclable such as polyurethane. 
            Avoid it if you can – recyclers generally speaking don’t want it.''')
}


def find_icons(image_path):
    matches = {}
    image = prepare_image(cv2.imread(image_path))
    for icon_name, icon in ICONS.items():
        template = prepare_template(cv2.imread(icon.image_path))
        match = find_template(image, template)
        print(icon_name, match)
        if match is not None and match.confidence > DETECTION_THRESHOLD:
            matches[icon_name] = match
    return matches
