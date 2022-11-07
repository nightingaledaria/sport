/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs';
import { math } from '@tensorflow/tfjs';

const color = 'aqua';
const boundingBoxColor = 'red';
const lineWidth = 2;

export const tryResNetButtonName = 'tryResNetButton';
export const tryResNetButtonText = '[New] Try ResNet50';
const tryResNetButtonTextCss = 'width:100%;text-decoration:underline;';
const tryResNetButtonBackgroundCss = 'background:#e61d5f;';

function isAndroid() {
    return /Android/i.test(navigator.userAgent);
}

function isiOS() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export function isMobile() {
    return isAndroid() || isiOS();
}

function setDatGuiPropertyCss(propertyText, liCssString, spanCssString = '') {
    var spans = document.getElementsByClassName('property-name');
    for (var i = 0; i < spans.length; i++) {
        var text = spans[i].textContent || spans[i].innerText;
        if (text == propertyText) {
            spans[i].parentNode.parentNode.style = liCssString;
            if (spanCssString !== '') {
                spans[i].style = spanCssString;
            }
        }
    }
}

export function updateTryResNetButtonDatGuiCss() {
    setDatGuiPropertyCss(
        tryResNetButtonText, tryResNetButtonBackgroundCss,
        tryResNetButtonTextCss);
}

/**
 * Toggles between the loading UI and the main canvas UI.
 */
export function toggleLoadingUI(
    showLoadingUI, loadingDivId = 'loading', mainDivId = 'main') {
    if (showLoadingUI) {
        document.getElementById(loadingDivId).style.display = 'block';
        document.getElementById(mainDivId).style.display = 'none';
    } else {
        document.getElementById(loadingDivId).style.display = 'none';
        document.getElementById(mainDivId).style.display = 'block';
    }
}

function toTuple({ y, x }) {
    return [y, x];
}

export function drawPoint(ctx, y, x, r, color) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

/**
 * Draws a line on a canvas, i.e. a joint
 */
export function drawSegment([ay, ax], [by, bx], color, scale, ctx) {
    ctx.beginPath();
    ctx.moveTo(ax * scale, ay * scale);
    ctx.lineTo(bx * scale, by * scale);
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;
    ctx.stroke();
}

/**
 * Draws a pose skeleton by looking up all adjacent keypoints/joints
 */
export function drawSkeleton(keypoints, minConfidence, ctx, scale = 1) {
    const adjacentKeyPoints =
        posenet.getAdjacentKeyPoints(keypoints, minConfidence);

    adjacentKeyPoints.forEach((keypoints) => {
        drawSegment(
            toTuple(keypoints[0].position), toTuple(keypoints[1].position), color,
            scale, ctx);
    });
}

/**
 * Draw pose keypoints onto a canvas
 */
export function drawKeypoints(keypoints, minConfidence, ctx, scale = 1) {
    for (let i = 0; i < keypoints.length; i++) {
        const keypoint = keypoints[i];

        if (keypoint.score < minConfidence) {
            continue;
        }

        const { y, x } = keypoint.position;
        drawPoint(ctx, y * scale, x * scale, 3, color);
    }
}

/** we will compare Y for different parts and will consider them equal
 * if their difference constitutes less than some predefined percent*/
export function AreApproximatelyEqual(y1, y2, scale, accuracy) {

    console.log("y1: " + y1 + " y2: " + y2 + " scale: " + scale + " accuracy: " + accuracy);
    var difference = 2 * Math.abs(y1 - y2) / scale;
    console.log("difference: " + difference);
    return difference < accuracy;
}
/**
 * Based on keypoints coordinates (Y=0 in the top of the image) check that the person is ready to start a new series
 * we assume that
 * 1) the following body parts have been identified:
 * nose,
 * shoulders,
 * elbows,
 * wrists
 * 2) shoulders Y are approx the same and wrists Y are approx the same
 * 3) wrists Y > elbows Y > shoulders Y
 * 3) return toppest (smaller) Y
 */
export function checkNewPushUpSeries(keypoints, minConfidence) {
    var maxY = -1;
    var minY = 1000;

    var leftWristPart = keypoints[9];
    var rightWristPart = keypoints[10];

    var leftShoulderPart = keypoints[5];
    var rightShoulderPart = keypoints[6];
    var iShoulderHighestPositionY = Math.max(leftShoulderPart.position.y, rightShoulderPart.position.y);

    var leftElbowPart = keypoints[7];
    var rightElbowPart = keypoints[8];
    var iElbowLowestPositionY = Math.min(leftElbowPart.position.y, rightElbowPart.position.y);

    if (iShoulderHighestPositionY < iElbowLowestPositionY) {
        minY = (leftShoulderPart.position.y + rightShoulderPart.position.y) / 2;
        maxY = (leftElbowPart.position.y + rightElbowPart.position.y) / 2;
    }


    //[minY, maxY] = calculateMaxandMinY(keypoints, minConfidence);

    return [minY, maxY]
}

export function calculateMaxandMinY(keypoints, minConfidence) {

    var maxY = -1;
    var minY = 1000;

    for (let i = 5; i < keypoints.length; i++) {
        const keypoint = keypoints[i];

        if (keypoint.score < minConfidence) {
            continue;
        }
        if (keypoint.position.y > maxY) {
            maxY = keypoint.position.y;
        }
        if (keypoint.position.y < minY) {
            minY = keypoint.position.y;
        }


    }
    return [minY, maxY];
}

export function AreKeypointsIdentified(keypoints, minConfidence, indexes) {
    for (let i = 0; i < indexes.length; i++) {
        const keypoint = keypoints[indexes[i]];

        if (keypoint.score < minConfidence) {
            console.log(" Not identified: " + keypoint.part)
            return false;
        } else {
            console.log(" identified: " + keypoint.part + "x: " + keypoint.position.x + "y: " + keypoint.position.y);
        }
    }
    return true;
}
/**
 * shoulders are approx on the same level
 * elbows are approx on the same level
 * Wrist are approx on the same level
 * head is approx where shoulders and approx at the initial Y
 *  var nosePart = keypoints[0];

                console.log(nosePart.part);
                console.log(nosePart.position.y);


                /*
                5	leftShoulder
                6	rightShoulder
                7	leftElbow
                8	rightElbow
                9	leftWrist
                10	rightWrist

 */



export function checkThatPushUpsInUpperPosition(keypoints, minConfidence, previousMinY, previousMaxY) {
    var nosePart = keypoints[0];
    var leftShoulderPart = keypoints[5];
    var rightShoulderPart = keypoints[6];
    var leftElbowPart = keypoints[7];
    var rightElbowPart = keypoints[8];
    var leftWristPart = keypoints[9];
    var rightWristPart = keypoints[10];

    //shoulders were identified
    if (AreKeypointsIdentified(keypoints, minConfidence, [5, 6])) {
        //shoulders are approximately on the same level
        if (AreApproximatelyEqual(leftShoulderPart.position.y, rightShoulderPart.position.y, previousMaxY, 0.20)) {
            //shoulders are approximately where they were in the begining of exercise
            if (AreApproximatelyEqual((leftShoulderPart.position.y + rightShoulderPart.position.y) / 2, previousMinY, previousMaxY, 0.20)) {
                return true;
            }
        }
    }
    return false;



    //compare
}

/**
 * shoulders are approx on the same level where elbows
 * wrists are approx on the minimum initial level
 * shoulders are approx on the min Y
 * */
export function checkThatPushUpsInLowerPosition(keypoints, minConfidence, previousMaxY) {
    var nosePart = keypoints[0];
    var leftShoulderPart = keypoints[5];
    var rightShoulderPart = keypoints[6];
    var leftElbowPart = keypoints[7];
    var rightElbowPart = keypoints[8];
    var leftWristPart = keypoints[9];
    var rightWristPart = keypoints[10];


    if (IsPushUpsInLowerPosition(keypoints, minConfidence, previousMaxY, 5, 6)) {
        return true;
    }

    if (IsPushUpsInLowerPosition(keypoints, minConfidence, previousMaxY, 1, 2)) {
        return true;
    }

    if (IsPushUpsInLowerPosition(keypoints, minConfidence, previousMaxY, 5, 8)) {
        return true;
    }

    if (IsPushUpsInLowerPosition(keypoints, minConfidence, previousMaxY, 6, 7)) {
        return true;
    }


    /*
    //shoulders were identified
    if (AreKeypointsIdentified(keypoints, minConfidence, [5, 6])) {
        //shoulders are approximately on the same level
        if (AreApproximatelyEqual(leftShoulderPart.position.y, rightShoulderPart.position.y, previousMaxY, 0.20)) {
            //shoulders are approximately where they were in the begining of exercise
            if (AreApproximatelyEqual((leftShoulderPart.position.y + rightShoulderPart.position.y) / 2, previousMaxY, previousMaxY, 0.2)) {
                return true;
            }
        }
    }

    if (AreKeypointsIdentified(keypoints, minConfidence, [1, 2])) {
      var leftEyePart = keypoints[1];
      var rightEyePart = keypoints[2];
      //shoulders are approximately on the same level
      if (AreApproximatelyEqual(leftEyePartposition.y, rightEyePart.position.y, previousMaxY, 0.20)) {
          //shoulders are approximately where they were in the begining of exercise
          if (AreApproximatelyEqual((leftEyePart.position.y + leftEyePart.position.y) / 2, previousMaxY, previousMaxY, 0.3)) {
              return true;
          }
      }

  } */
    return false;
}

export function IsPushUpsInLowerPosition(keypoints, minConfidence, previousMaxY, firstPartIndex, secondPartIndex) {

    var firstPart = keypoints[firstPartIndex];
    var secondPart = keypoints[secondPartIndex];


    //requested parts were identified
    if (AreKeypointsIdentified(keypoints, minConfidence, [firstPartIndex, secondPartIndex])) {
        //requested parts are approximately on the same level
        if (AreApproximatelyEqual(firstPart.position.y, secondPart.position.y, previousMaxY, 0.20)) {
            //shoulders are approximately where they were in the begining of exercise
            if (AreApproximatelyEqual((firstPart.position.y + secondPart.position.y) / 2, previousMaxY, previousMaxY, 0.2)) {
                return true;
            }
        }
    }
    return false;
}

/**
 * Draw the bounding box of a pose. For example, for a whole person standing
 * in an image, the bounding box will begin at the nose and extend to one of
 * ankles
 */
export function drawBoundingBox(keypoints, ctx) {
    const boundingBox = posenet.getBoundingBox(keypoints);

    ctx.rect(
        boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX,
        boundingBox.maxY - boundingBox.minY);

    ctx.strokeStyle = boundingBoxColor;
    ctx.stroke();
}

/**
 * Converts an arary of pixel data into an ImageData object
 */
export async function renderToCanvas(a, ctx) {
    const [height, width] = a.shape;
    const imageData = new ImageData(width, height);

    const data = await a.data();

    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        const k = i * 3;

        imageData.data[j + 0] = data[k + 0];
        imageData.data[j + 1] = data[k + 1];
        imageData.data[j + 2] = data[k + 2];
        imageData.data[j + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
}

/**
 * Draw an image on a canvas
 */
export function renderImageToCanvas(image, size, canvas) {
    canvas.width = size[0];
    canvas.height = size[1];
    const ctx = canvas.getContext('2d');

    ctx.drawImage(image, 0, 0);
}

/**
 * Draw heatmap values, one of the model outputs, on to the canvas
 * Read our blog post for a description of PoseNet's heatmap outputs
 * https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
 */
export function drawHeatMapValues(heatMapValues, outputStride, canvas) {
    const ctx = canvas.getContext('2d');
    const radius = 5;
    const scaledValues = heatMapValues.mul(tf.scalar(outputStride, 'int32'));

    drawPoints(ctx, scaledValues, radius, color);
}

/**
 * Used by the drawHeatMapValues method to draw heatmap points on to
 * the canvas
 */
function drawPoints(ctx, points, radius, color) {
    const data = points.buffer().values;

    for (let i = 0; i < data.length; i += 2) {
        const pointY = data[i];
        const pointX = data[i + 1];

        if (pointX !== 0 && pointY !== 0) {
            ctx.beginPath();
            ctx.arc(pointX, pointY, radius, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
        }
    }
}

/**
 * Draw offset vector values, one of the model outputs, on to the canvas
 * Read our blog post for a description of PoseNet's offset vector outputs
 * https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
 */
export function drawOffsetVectors(
    heatMapValues, offsets, outputStride, scale = 1, ctx) {
    const offsetPoints =
        posenet.singlePose.getOffsetPoints(heatMapValues, outputStride, offsets);

    const heatmapData = heatMapValues.buffer().values;
    const offsetPointsData = offsetPoints.buffer().values;

    for (let i = 0; i < heatmapData.length; i += 2) {
        const heatmapY = heatmapData[i] * outputStride;
        const heatmapX = heatmapData[i + 1] * outputStride;
        const offsetPointY = offsetPointsData[i];
        const offsetPointX = offsetPointsData[i + 1];

        drawSegment(
            [heatmapY, heatmapX], [offsetPointY, offsetPointX], color, scale, ctx);
    }
}
export function drawNumberOfPushes(iNumOfPushes) {
    document.getElementById("NumberOfPushes").innerHTML = "" + iNumOfPushes;
}