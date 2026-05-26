#import "@preview/diatypst:0.7.1": *

#show: slides.with(
  title: [swarm alarm 🐝],
  subtitle: [CVAI - homework],
  date: datetime.today().display("[day padding:zero]. [month repr:short]. [year repr:full]"),
  ratio: 16 / 9,
  title-color: rgb("#ffcc00"),
  toc: false,
  authors: "Dominic Bieri, Elias Christen, Laura Grüter",
)

= Problem to be Solved
== Problem to be Solved
- When a bee colony decides to swarm, the queen leaves the hive unexpectedly, along with thousands of worker bees.

- After leaving, the swarm gathers in the immediate vicinity. The beekeeper often has only a few hours to capture the swarm.

- For the beekeeper, this means the potential loss of half the colony and the associated honey production.

#figure(
  image("../media/presentation/image-3.png", width: 20%),
  caption: [
    #set text(size: 0.7em)
    bee swarm, Wikipedia
  ],
  supplement: none,
)
= Result and Demo
== Result
- Camera installation and Tapo Stream
- Image labeling with Label Studio
- Train CNN with YOLO
- #link("https://swarm-alarm.crstn.ch/")[Swarm alarm application]
- #link("https://github.com/dominic-bieri/cvai-homework")[GitHub Repository]

#image("../media/presentation/image-7.png", width: 30%)

= Path - First Try
== First Try
- Existing dataset of bees from Roboflow
- Existing camera installation
- Model training with YOLO
- Well...

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  image("../media/presentation/image-4.png"), image("../media/presentation/image-5.png"),
)

= Improvements
== Camera Installation - Closer to the Bees

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  image("/media/presentation/image-19.png", width: 60%), image("/media/presentation/image-22.png", width: 80%),
)

Christenbaum is watching you...
== Manually Labeled Data with Label Studio
- Manually labeled images (screenshots from Tapo stream at different times / lighting conditions)
- Using bounding box for "bee" classification

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  image("../media/presentation/image-8.png", width: 60%), image("../media/presentation/image-9.png", width: 60%),
  image("../media/presentation/image-10.png", width: 60%), image("../media/presentation/image-11.png", width: 60%),
)

= Result
== Result
Bee detection getting better and better:
#image("../media/presentation/image-15.png", width: 40%)

And these "Saublüemli" are no longer bees:
#image("../media/presentation/image-13.png", width: 20%)

== Snow
Some unexpected weather conditions....

#image("../media/presentation/image-14.png", width: 50%)

The bees that were detected weren't bees - so we need more images.

= Final Result and Technical Background

== Train CNN
- Manually labeled images in Label Studio
- Create dataset (train, validation, test)
- Data augmentation
  - augment only the training data
- Train the model YOLO26n
- "Active learning"
  - get faster more labled data
  - captures new frames from the camera and sends them to Label Studio along with pre-annotations
  - manuel step required

== Data Augmentation using albumentationsx
#show raw: set text(size: 7pt)
```python
pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(angle_range=(-30, 30), p=0.5),
        A.RandomBrightnessContrast(brightness_range=(-0.3, 0.3), contrast_range=(-0.3, 0.3), p=0.6),
        A.HueSaturationValue(hue_shift_range=(-10, 10), sat_shift_range=(-30, 30), val_shift_range=(-20, 20), p=0.4),
        A.RandomShadow(p=0.3),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.1),
        A.MotionBlur(blur_range=(3, 7), p=0.3),
        A.GaussianBlur(blur_range=(3, 5), p=0.2),
        A.GaussNoise(p=0.3),
        A.ImageCompression(quality_range=(60, 100), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(20, 60), hole_width_range=(20, 60), p=0.3),
    ],
    bbox_params=A.BboxParams(
        coord_format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
    ),
)
```

== Conclusion
- Not all bees are detected correctly
- But for our use case it is not relevant
  - We need to know when the bees swarm and not an exact number of bees
- In another environment it would probably not work - manually labeled data from the new environment are needed
- Tracking: It could be very useful for our use case
  - We did some testing with tracking


#image("../media/presentation/image-16.png", width: 40%)

== Have you ever seen a beekeeper at work?
You may spot one, if you are lucky :)


#image("../media/presentation/image-18.png", width: 60%)

= Questions and thanks for your attention
