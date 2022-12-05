# sample-read-write-transform

## call event to image serialization

```console
python3 -u event_to_image_serialization.py -in input_dir/input_file.h5 -out output_dir/output_file.h5
```

with optional limitation on number of events:

```console
-n 10000
```

## call concatenate events serialization

```console
python3 -u event_concatenate_serialization.py -in input_dir -out output_dir/output_file.h5
```
with optional limitation on number of events to read from input directory:

```console
-n 10000
```

with optional limitation on file size in [MB] (in this case, output is split into multiple files indexed by _i part flag):

```console
-mb 100
```
