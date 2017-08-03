# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description: SketchRNN generative model of sketches.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "sketch_rnn",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":model",
        ":rnn",
        ":utils",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":rnn",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "rnn",
    srcs = ["rnn.py"],
    deps = [
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        # numpy dep
    ],
)

py_binary(
    name = "sketch_rnn_train",
    srcs = ["sketch_rnn_train.py"],
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":model",
        ":utils",
        # numpy dep
        # tensorflow dep
    ],
)
