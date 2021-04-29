// $(document).ready(function () {
//   $.ajaxSetup({cache: false, headers: {"cache-control": "no-cache"}});
// });

// Set the SVG height and width
function svg_setup() {
  let container_dims = document.getElementById('scatter-holder').getBoundingClientRect();
  let svg = d3.select('#animation-svg');
  let svg_dim = Math.min(container_dims.width, container_dims.height);
  svg.attr('width', svg_dim)
    .attr('height', svg_dim)
}

svg_setup()
d3.select(window).on('resize', svg_setup);

// Fill the textboxes while testing
let TESTING = true;

// Initialize global variables
let LABEL_VISIBILITY = true;
let MEAN_VISIBILITY = true;
let EVAL_VISIBILITY = true;
let REMOVE_POINTS = false;
let ANIMSTEP_COUNTER = 0;
let ANIMATION_DURATION = 4000;
let AXIS_TOLERANCE = 0.05;
let INTERPOLATION = d3.easeLinear;
let DYNAMIC_PROJ = false;
let zoom = null;

if (TESTING) {
  ANIMATION_DURATION = 500;
}

let labelArray = [];
let anchorArray = [];

// Set global color-scale
let color = d3.scaleOrdinal(d3.schemeDark2);
let shape = d3.scaleOrdinal([0, 1, 2, 3, 4, 5, 6], [d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolSquare, d3.symbolTriangle, d3.symbolCross].map(d => symbolGenerator(d)));

function symbolGenerator(symbolObj) {
  return d3.symbol().type(symbolObj).size(125)();
}

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function filterByName(name, array, prop) {
  return array.filter(d => d[prop] === name)[0];
}

function draw_scatter(svg, response, x, y, width, height, margin) {
  let point_data = response.emb1.concat(response.emb2);

  // Add the scatterplot
  let defs = svg.append('defs');
  defs.append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', [0, 0, 10, 10])
    .attr('refX', '5')
    .attr('refY', '5')
    .attr('markerWidth', '3')
    .attr('markerHeight', '3')
    .attr('stroke', '#5b5b5b')
    .attr('orient', 'auto-start-reverse')
    .append('path')
    .attr('d', 'M 0 0 L 10 5 L 0 10 z');

  let gradient = defs.append('linearGradient')
    .attr('id', 'gradient');

  gradient.append('stop')
    .attr('offset', '0%')
    .attr('stop-color', 'red');

  gradient.append('stop')
    .attr('offset', '100%')
    .attr('stop-color', 'blue');


  labelArray = [];
  anchorArray = [];

  let datapoint_group = svg.selectAll('g')
    .data(point_data)
    .enter()
    .append('g')
    .attr('class', d => 'datapoint-group group-' + d.group)
    .attr('transform', d => {
      let x_coord = x(d.x), y_coord = y(d.y)
      labelArray.push({x: x_coord, y: y_coord, name: d.label, group: d.group})
      anchorArray.push({x: x_coord, y: y_coord, r: 10})
      return 'translate(' + x(d.x) + ',' + y(d.y) + ')'
    })
    .on('mouseover', function () {
      svg.selectAll('g.datapoint-group').classed('translucent', true);
      d3.select(this).classed('translucent', false);
    })
    .on('mouseout', function () {
      svg.selectAll('g.datapoint-group').classed('translucent', false);
    })

  // Draw labels
  let labels = datapoint_group.append('text')
    .text((d, i) => d.group === 0 ? '' : labelArray[i].name)
    .attr('dx', (d, i) => labelArray[i].x - anchorArray[i].x)
    .attr('dy', (d, i) => labelArray[i].y - anchorArray[i].y)
    .classed('class-label', 'true')
    .attr('fill', (d, i) => labelArray[i].group === 0 ? 'black' : color(labelArray[i].group))

  // Size of each label
  let index = 0;
  labels.each(function () {
    labelArray[index].width = this.getBBox().width;
    labelArray[index].height = this.getBBox().height;
    index += 1;
  });

  d3.labeler()
    .label(labelArray)
    .anchor(anchorArray)
    .width(width)
    .height(height)
    .start(2000);

  labels.transition()
    .duration(500)
    .attr('x', (d, i) => labelArray[i].x - anchorArray[i].x)
    .attr('y', (d, i) => labelArray[i].y - anchorArray[i].y)

  datapoint_group.append('path')
    .attr('fill', d => d.group === 0 ? '#414141' : color(d.group))
    .attr('d', d => (d.group === 0 && d.label != 'Origin') ? '' : shape(d.group))
    .attr('stroke', 'black')
    .attr('stroke-width', '1px')
    .attr('stroke-opacity', '0.75')

  // Draw the bias direction arrow
  let arrow_endpoints = [point_data.filter(d => d.group === 0).map(d => [d.x, d.y])];

  zoom = d3.zoom().scaleExtent([0.02, 20]).extent([[0, 0], [width, height]]).on("zoom", update_plot);

  svg.append('rect')
    .attr('width', width)
    .attr('height', height)
    .attr('fill', 'none')
    .attr('pointer-events', 'all')
    .lower();

  svg.call(zoom);

  function update_plot() {
    let newX = d3.event.transform.rescaleX(x);
    let newY = d3.event.transform.rescaleY(y);

    // update axes with these new boundaries
    svg.select('.x').call(d3.axisBottom(newX))
    svg.select('.y').call(d3.axisLeft(newY));

    datapoint_group.transition()
      .duration(0)
      .attr('transform', d => 'translate(' + newX(d.x) + ',' + newY(d.y) + ')');

    d3.select('#alignment-lines').attr('transform', d3.event.transform);
  }


}

function draw_alignment_lines(svg, response, x, y) {
  let line_data = Array();

  for (let i = 0; i < response.emb1.length; i++) {
    let start_point = [x(response.emb1[i].x), y(response.emb1[i].y)]
    let end_point = [x(response.emb2[i].x), y(response.emb2[i].y)]
    line_data.push([start_point, end_point])
  }

  svg.insert('g', 'defs')
    .attr('id', 'alignment-lines')
    .selectAll('path')
    .data(line_data)
    .join('path')
    .attr('d', d => d3.line()(d))
    // .attr('stroke', 'url(#gradient)')
    .attr('stroke', 'black')
    .attr('stroke-width', '1px')
    .classed('align-lines', true)
  // .attr('stroke-dasharray', '5, 5');
}

function draw_axes(svg, width, height, x, y) {
  // Add the X Axis
  let x_axis = svg.append('g')
    .attr('transform', 'translate(0,' + height + ')')
    .classed('x axis', true)
    .call(d3.axisBottom(x));

  // Add the Y Axis
  let y_axis = svg.append('g')
    .classed('y axis', true)
    .call(d3.axisLeft(y));

  return [x_axis, y_axis];
}

function compute_axes_limits_sym(response) {
  let points = response.emb1.concat(response.emb2);
  let x_coords = points.map(d => Math.abs(d.x));
  let y_coords = points.map(d => Math.abs(d.y));
  let x = Math.max(...x_coords), y = Math.max(...y_coords);
  let coord_max = Math.max(x, y);
  return {
    x_min: -coord_max - 0.2 * coord_max, x_max: coord_max + 0.2 * coord_max,
    y_min: -coord_max - 0.2 * coord_max, y_max: coord_max + 0.2 * coord_max
  }
}

function compute_perpendicular(line) {
  let x = line[0], y = line[1];
  if (Math.abs(x) <= 0.0000001) {
    return [0, 0]
  }
  return [-line[1] / line[0], 1];
}

function setup_drawing(anim_svg, response, identifier) {
  try {

    let margin = {top: 20, right: 20, bottom: 20, left: 40};
    let width = anim_svg.node().width.baseVal.value - margin.left - margin.right;
    let height = anim_svg.node().height.baseVal.value - margin.top - margin.bottom;

    // set the ranges
    let x_axis = d3.scaleLinear().range([0, width - 30]);
    let y_axis = d3.scaleLinear().range([height, 0]);

    let axes_limits = compute_axes_limits_sym(response);
    console.log(response)
    x_axis.domain([axes_limits['x_min'], axes_limits['x_max']]).nice();
    y_axis.domain([axes_limits['y_min'], axes_limits['y_max']]).nice();


    let svg = anim_svg.append('g')
      .attr('id', identifier + 'group')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    draw_scatter(svg, response, x_axis, y_axis, width, height, margin);
    draw_alignment_lines(svg, response, x_axis, y_axis);
    draw_axes(svg, width, height, x_axis, y_axis);
  } catch (e) {
    console.log(e);
  }
}

function btn_active(btn, bool_active) {
  btn.prop('disabled', !bool_active);
}

function svg_cleanup() {
  $('#pre-debiased-svg').empty();
  $('#animation-svg').empty();
  $('#post-debiased-svg').empty();
}

$('#emb1 a').click(function (e) {
  try {
    let emb1 = this.innerHTML;
    $('#emb1-button').text(emb1);
  } catch (e) {
    console.log(e);
  }
});

$('#emb2 a').click(function (e) {
  try {
    let emb2 = this.innerHTML;
    $('#emb2-button').text(emb2);
  } catch (e) {
    console.log(e);
  }
});

$('#emb1-chk').click(function (e) {
  if (e.target.checked === true) {
    d3.selectAll('.group-1').attr('visibility', 'visible');
    d3.selectAll('.align-lines').attr('visibility', 'visible');
  } else {
    d3.selectAll('.group-1').attr('visibility', 'hidden');
    d3.selectAll('.align-lines').attr('visibility', 'hidden');
  }
})

$('#emb2-chk').click(function (e) {
  if (e.target.checked === true) {
    d3.selectAll('.group-2').attr('visibility', 'visible');
    d3.selectAll('.align-lines').attr('visibility', 'visible');
  } else {
    d3.selectAll('.group-2').attr('visibility', 'hidden');
    d3.selectAll('.align-lines').attr('visibility', 'hidden');
  }
})

// Functionality for the 'Run' button
$('#run-button').click(function () {
  try { // Perform cleanup
    svg_cleanup();

    let emb1 = $('#emb1-button').text();
    let emb2 = $('#emb2-button').text();
    let wordlist = $('#wordlist').val();
    let xwords = $('#dir1').val();
    let ywords = $('#dir2').val();

    $.ajax({
      type: 'POST',
      url: '/align_embs',
      data: {
        emb1: emb1,
        emb2: emb2,
        wordlist: wordlist,
        xwords: xwords,
        ywords: ywords
      },
      // beforeSend: function () {
      //   $('.overlay').addClass('d-flex').show();
      //   $('#spinner-holder').show();
      // },
      success: function (response) {
        let animation_svg = d3.select('#animation-svg');
        setup_drawing(animation_svg, response, 'animation')
      },
      // complete: function () {
      //   $('.overlay').removeClass('d-flex').hide();
      //   $('#spinner-holder').hide();
      // },
      error: function (request, status, error) {
        alert(request.responseJSON.message);
      }
    });
  } catch (e) {
    console.log(e);
  }
});

if (TESTING) {
  try {
    $('#wordlist').val('he, him, she, her, father, mother, doctor, banker, nurse, engineer');
    $('#dir1').val('he, him, she, her');
    $('#dir2').val('doctor, banker, nurse, maid, engineer');
    $('#emb1-items').children()[0].click();
    $('#emb2-items').children()[2].click();
    $('#run-button').click();
  } catch (e) {
    console.log(e);
  }
}
