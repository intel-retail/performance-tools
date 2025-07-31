#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

# Set default resolution if not provided
WIDTH="${1:-1920}"
HEIGHT="${2:-1080}"

./format_avc_mp4.sh coca-cola-4465029.mp4 https://www.pexels.com/download/video/4465029 "$1" "$2" "$3"

./format_avc_mp4.sh coca-cola-4465029.mp4 https://www.pexels.com/download/video/4465029 "$WIDTH" "$HEIGHT" "25"

./format_avc_mp4.sh supermarket_1.mp4 https://www.pexels.com/download/video/3249935 "$WIDTH" "$HEIGHT" "15"

# up to 3 bottles and human hand
#./format_avc_mp4.sh vehicle-bike.mp4 https://www.pexels.com/download/video/853908 "$1" "$2" "$3"
#./format_avc_mp4.sh group-of-friends-smiling-3248275.mp4 https://www.pexels.com/download/video/3248275 "$1" "$2" "$3"
#./format_avc_mp4.sh grocery-items-on-the-kitchen-shelf-4983686.mp4 https://www.pexels.com/video/4983686/download/ $1 $2 $3
#./format_avc_mp4.sh video_of_people_walking_855564.mp4 https://www.pexels.com/download/video/855564 "$1" "$2" "$3"
#./format_avc_mp4.sh barcode.mp4 https://github.com/antoniomtz/sample-clips/raw/main/barcode.mp4 "$1" "$2" "$3"
#./format_avc_mp4.sh vehicle-bike.mp4 https://www.pexels.com/download/video/853908 "$1" "$2" "$3"
#./format_avc_mp4.sh group-of-friends-smiling-3248275.mp4 https://www.pexels.com/download/video/3248275 "$1" "$2" "$3"
#./format_avc_mp4.sh grocery-items-on-the-kitchen-shelf-4983686.mp4 https://www.pexels.com/video/4983686/download/ $1 $2 $3
#./format_avc_mp4.sh video_of_people_walking_855564.mp4 https://www.pexels.com/download/video/855564 "$1" "$2" "$3"
#./format_avc_mp4.sh barcode.mp4 https://github.com/antoniomtz/sample-clips/raw/main/barcode.mp4 "$1" "$2" "$3"
