"use strict";

// lets do some fun
var video = document.getElementById('webcam');
var canvas = document.getElementById('canvas');
var attempts = 0;

var readyListener = function(event) {
    findVideoSize();
};
var findVideoSize = function() {
    if(video.videoWidth > 0 && video.videoHeight > 0) {
        video.removeEventListener('loadeddata', readyListener);
        onDimensionsReady(video.videoWidth, video.videoHeight);
    } else {
        if(attempts < 10) {
            attempts++;
            setTimeout(findVideoSize, 200);
        } else {
            onDimensionsReady(640, 480);
        }
    }
};
var onDimensionsReady = function(width, height) {
    start(width, height);
    requestAnimationFrame(tick);
};

video.addEventListener('loadeddata', readyListener);

navigator.getUserMedia({video: true}, function(stream) {
    try {
        video.src = window.URL.createObjectURL(stream);
    } catch (error) {
        video.src = stream;
    }
    setTimeout(function() {
            video.play();
        }, 500);
}, function (error) {
    console.error(error);
});


var ctx,canvasWidth,canvasHeight;
var curr_img_pyr, prev_img_pyr, point_count, point_status, prev_xy, curr_xy;
var selectedPoints = [];
var maxPoints = 200;
var options = {
    win_size: 20,
    max_iterations: 30,
    epsilon: 0.01,
    min_eigen: 0.001,
}
function start(videoWidth, videoHeight) {
    canvasWidth  = canvas.width;
    canvasHeight = canvas.height;
    ctx = canvas.getContext('2d');

    curr_img_pyr = new jsfeat.pyramid_t(3);
    prev_img_pyr = new jsfeat.pyramid_t(3);
    curr_img_pyr.allocate(640, 480, jsfeat.U8_t|jsfeat.C1_t);
    prev_img_pyr.allocate(640, 480, jsfeat.U8_t|jsfeat.C1_t);

    point_count = 0;
    point_status = new Uint8Array(maxPoints);
    prev_xy = new Float32Array(maxPoints*2);
    curr_xy = new Float32Array(maxPoints*2);

    setTimeout(randomizeBrutal, 1000);
    //setInterval(randomizeBrutal, 1000);
}

function randomizeBrutal() {
    var numPointsX = 10;
    var numPointsY = 10;
    var offsetX = canvasWidth / (numPointsX+1);
    var offsetY = canvasHeight / (numPointsY+1);
    for(var x = offsetX/2; x < canvasWidth; x += offsetX) {
        for(var y = offsetY/2; y < canvasHeight; y += offsetY) {
            addPoint(Math.ceil(x), Math.ceil(y));
        }
    }
}
function tick() {
    requestAnimationFrame(tick);
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        ctx.drawImage(video, 0, 0, 640, 480);
        var imageData = ctx.getImageData(0, 0, 640, 480);

        // swap flow data
        var _pt_xy = prev_xy;
        prev_xy = curr_xy;
        curr_xy = _pt_xy;
        var _pyr = prev_img_pyr;
        prev_img_pyr = curr_img_pyr;
        curr_img_pyr = _pyr;

        jsfeat.imgproc.grayscale(imageData.data, 640, 480, curr_img_pyr.data[0]);

        curr_img_pyr.build(curr_img_pyr.data[0], true);

        jsfeat.optical_flow_lk.track(prev_img_pyr, curr_img_pyr, prev_xy, curr_xy, point_count, options.win_size|0, options.max_iterations|0, point_status, options.epsilon, options.min_eigen);
        removeDoubles();
        prune_oflow_points(ctx);

        checkSelection();
        const oldSelection = JSON.parse(JSON.stringify(selectedPoints));
        selectedPoints = selectedPoints.map(point => getPointAt(point.idx));

        const motion = getMotion(oldSelection, selectedPoints);
        //console.log(JSON.stringify(oldSelection), JSON.stringify(selectedPoints));
        const debug = document.getElementById('debug').innerHTML = `
            ${ motion.data[0] }, ${ motion.data[1] }, ${ motion.data[2] }<br>
            ${ motion.data[3] }, ${ motion.data[4] }, ${ motion.data[5] }<br>
            ${ motion.data[6] }, ${ motion.data[7] }, ${ motion.data[8] }<br>
        `;
        console.log(debug);
    }
}
function checkSelection() {
    selectedPoints.filter(point => point_status[point.idx] === 1);
}

// remove points when they are too close
// uses a system of tile to detect "collisions"
function removeDoubles() {
    const TILE_SIZE = 10;
    const tiles = [];
    const doubles = [];
    for(let idx = 0; idx < point_count; idx++) {
        const point = getPointAt(idx);
        if(point) {
            const tileX = Math.round(point.x/TILE_SIZE);
            const tileY = Math.round(point.y/TILE_SIZE);
            const tile = tiles.find(tile => tile.x === tileX && tile.y === tileY);
            if(tile) {
                doubles.push(point);
            }
            else tiles.push({x: tileX, y: tileY});
        }
    }
    doubles.forEach(point => removePointAt(point.idx));
}

function addPoint(x, y) {
    const idx = point_status.findIndex(status => status === 0);
    if(idx > -1 && idx < point_count) {
        curr_xy[idx<<1] = x;
        curr_xy[(idx<<1)+1] = y;
        point_status[idx] = 1;
    }
    else if(point_count<=maxPoints) {
        curr_xy[point_count<<1] = x;
        curr_xy[(point_count<<1)+1] = y;
        point_status[point_count] = 1;
        point_count++;
    }
    else {
        console.warn('could not add a point, max points alload');
    }
}

function removePointAt(idx) {
    point_count--;
    point_status[idx] = 0;
    curr_xy[idx<<1] = 0;
    curr_xy[(idx<<1)+1] = 0;
}

function on_canvas_click(e) {
    var coords = canvas.relMouseCoords(e);
    if(coords.x > 0 & coords.y > 0 & coords.x < canvasWidth & coords.y < canvasHeight) {
        selectedPoints = getClosestPoints(coords.x, coords.y, 4);
        console.log('points:', selectedPoints.map(point => point.distance));
    }
}
function getMotion(srcPoints, dstPoints) {
    const matrix = new jsfeat.matrix_t(3, 3, jsfeat.U8_t|jsfeat.C1_t);
    if(srcPoints.length === 4 || dstPoints.length === 4) {
        jsfeat.math.perspective_4point_transform(matrix, 
            srcPoints[0].x, srcPoints[0].y, dstPoints[0].x, dstPoints[0].y,
            srcPoints[1].x, srcPoints[1].y, dstPoints[1].x, dstPoints[1].y,
            srcPoints[2].x, srcPoints[2].y, dstPoints[2].x, dstPoints[2].y,
            srcPoints[3].x, srcPoints[3].y, dstPoints[3].x, dstPoints[3].y);
    }
    return matrix;
}
function getPointAt(idx) {
    return point_status[idx] === 0 || !curr_xy[idx<<1] ? null : {
        x: curr_xy[idx<<1],
        y: curr_xy[(idx<<1)+1],
        idx: idx
    };
}
function getClosestPoints(x, y, num) {
    const res = [];
    for(let idx = 0; idx < point_count; idx++) {
        const point = getPointAt(idx);
        if(point) {
            res.push(point);
            point.distance = Math.hypot(x - point.x, y - point.y);
        }
    }
    return res.sort((point1, point2) => point1.distance - point2.distance).slice(0, num);
}

canvas.addEventListener('click', on_canvas_click, false);

function draw_circle(ctx, x, y, diam, fillStyle, strokeStyle) {
    ctx.fillStyle = fillStyle || "rgb(0,255,0)";
    ctx.strokeStyle = strokeStyle || "rgb(0,255,0)";

    ctx.beginPath();
    ctx.arc(x, y, diam, 0, Math.PI*2, true);
    ctx.closePath();
    ctx.fill();
}
function isSelected(i) {
    return selectedPoints.findIndex(point => !!point && point.idx === i) >= 0;
}
function prune_oflow_points(ctx) {
    var n = point_count;
    var i=0,j=0;

    for(; i < n; ++i) {
        if(point_status[i] == 1) {
            if(j < i) {
                curr_xy[j<<1] = curr_xy[i<<1];
                curr_xy[(j<<1)+1] = curr_xy[(i<<1)+1];
            }
            draw_circle(ctx, curr_xy[j<<1], curr_xy[(j<<1)+1], isSelected ? 6 : 4, isSelected(i) ? "rgb(255, 0, 0)" : null);
            ++j;
        }
    }
    point_count = j;
}

function relMouseCoords(event) {
    console.log(event)

    var totalOffsetX=0,totalOffsetY=0,canvasX=0,canvasY=0;
    var currentElement = this;

    do {
        totalOffsetX += currentElement.offsetLeft - currentElement.scrollLeft;
        totalOffsetY += currentElement.offsetTop - currentElement.scrollTop;
    } while(currentElement = currentElement.offsetParent)

    canvasX = event.pageX - totalOffsetX;
    canvasY = event.pageY - totalOffsetY;

    return {x:canvasX, y:canvasY}
}
HTMLCanvasElement.prototype.relMouseCoords = relMouseCoords;

window.onunload = function() {
    video.pause();
    video.src=null;
};

