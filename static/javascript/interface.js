// $(document).ready(function () {
//   $.ajaxSetup({cache: false});
// });

// Fill the textboxes while testing
let TESTING = false;

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
  ANIMATION_DURATION = 2000;
}

let labelArray = [];
let anchorArray = [];

let ALGO_MAP = {
  'Linear projection': 1,
  'Hard debiasing': 2,
  'OSCaR': 3,
  'Iterative Null Space Projection': 4
}

let SUBSPACE_MAP = {
  'Two means': 1,
  'PCA': 2,
  'PCA-paired': 3,
  'Classification': 4,
  'GSS': 5
}

// Set global color-scale
let color = d3.scaleOrdinal(d3.schemeDark2);
let shape = d3.scaleOrdinal([0, 1, 2, 3, 4, 5, 6],
  [d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolSquare, d3.symbolTriangle, d3.symbolCross].map(d => symbolGenerator(d)));

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

function process_response(response, eval = false, debiased = false) {
  let data = [];
  let vec1key = debiased ? 'debiased_vectors1' : 'vectors1';
  let vec2key = debiased ? 'debiased_vectors2' : 'vectors2';
  let debiased_veckey = debiased ? 'debiased_evalvecs' : 'evalvecs'

  for (let i = 0; i < response[vec1key].length; i++) {
    data.push({'position': response[vec1key][i], 'label': response['words1'][i], 'group': 1});
  }

  for (let i = 0; i < response[vec2key].length; i++) {
    data.push({'position': response[vec2key][i], 'label': response['words2'][i], 'group': 2});
  }

  if (eval) {
    for (let i = 0; i < response[debiased_veckey].length; i++) {
      data.push({'position': response[debiased_veckey][i], 'label': response.evalwords[i], 'group': 3});
    }
  }

  return data;
}

function vector_mean(arr) {
  let mean = [0, 0];
  let i;
  for (i = 0; i < arr.length; i++) {
    mean[0] += arr[i][0];
    mean[1] += arr[i][1];
  }
  mean[0] /= i;
  mean[1] /= i;
  return mean;
}

function check_if_mean(datapoint) {
  if (datapoint.label.startsWith('mean')) {
    return 1
  }
  return 0;
}

function remove_point() {
  let element = d3.select(this);
  let label = element.datum().label;
  let group = element.datum().group;
  if (group === 1) {
    let seedwords = $('#seedword-text-1').val().split(', ');
    let filtered = seedwords.filter(elem => elem !== label);
    $('#seedword-text-1').val(filtered.join(', '));
  }
  if (group === 2) {
    let seedwords = $('#seedword-text-2').val().split(', ');
    let filtered = seedwords.filter(elem => elem !== label);
    $('#seedword-text-2').val(filtered.join(', '));
  }
  if (group === 3) {
    let evalwords = $('#evaluation-list').val().split(', ');
    let filtered = evalwords.filter(elem => elem !== label);
    $('#evaluation-list').val(filtered.join(', '));
  }
  if (group === 4) {
    let orth_subspace_words = $('#oscar-seedword-text-1').val().split(', ');
    let filtered = orth_subspace_words.filter(elem => elem !== label);
    $('#oscar-seedword-text-1').val(filtered.join(', '));
  }
  console.log(element)
  element.attr('visibility', 'hidden')
  d3.select(this.parentNode).attr('visibility', 'hidden');
}

function sample_label_position(scale) {
  let pos = Math.random() + 1;
  return parseInt(pos * scale);
}

function draw_scatter_static(parent_svg, response, plotTitle, debiased = false) {
  p_svg = parent_svg;
  parent_svg.selectAll('*').remove();

  let margin = {top: 20, right: 20, bottom: 20, left: 40};
  let width = parent_svg.node().width.baseVal.value - margin.left - margin.right;
  let height = parent_svg.node().height.baseVal.value - margin.top - margin.bottom;

  // Append group to the svg
  let svg = parent_svg
    .append('g')
    .attr('id', plotTitle + 'group')
    .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');


  // set the ranges
  let x = d3.scaleLinear().range([0, width - 30]);
  let y = d3.scaleLinear().range([height, 0]);

  let axes_limits;
  if (debiased) {
    axes_limits = compute_axes_limits_sym(response.anim_steps[response.anim_steps.length - 1]);
  } else {
    axes_limits = compute_axes_limits_sym(response.anim_steps[0]);
  }

  x.domain([axes_limits['x_min'], axes_limits['x_max']]).nice();
  y.domain([axes_limits['y_min'], axes_limits['y_max']]).nice();

  let data = debiased ? response.debiased : response.base;

  // Add the scatterplot
  let datapoint_group = svg.selectAll('g')
    .data(data)
    .enter()
    .append('g')
    .attr('class', d => 'datapoint-group group-' + d.group)
    .attr('transform', d => 'translate(' + x(d.x) + ',' + y(d.y) + ')')
    .on('mouseover', function () {
      parent_svg.selectAll('g.datapoint-group').classed('translucent', true);
      parent_svg.select('#bias-direction-line').classed('translucent', true);
      d3.select(this).classed('translucent', false);
    })
    .on('mouseout', function () {
      parent_svg.selectAll('g.datapoint-group').classed('translucent', false);
      parent_svg.selectAll('#bias-direction-line').classed('translucent', false);
    })

  // Class label
  datapoint_group.append('foreignObject')
    .attr('x', 15)
    .attr('y', -10)
    .attr('width', '1px')
    .attr('height', '1px')
    .attr('class', 'fobj')
    .append('xhtml:div')
    .attr('class', 'class-label')
    .attr('style', d => 'color:' + (d.group === 0 ? 'black' : color(d.group)) + '; font-weight: 430; opacity:0.8; font-size: 1.2em')
    .html(d => d.label);


  // Remove buttons
  datapoint_group.append('text')
    .attr('class', 'fa cross-button')
    .attr('x', 10)
    .attr('y', -10)
    .attr('visibility', 'hidden')
    .on('click', remove_point)
    .text('\uf057');

  datapoint_group.append('path')
    .attr('fill', d => d.group === 0 ? '#414141' : color(d.group))
    .attr('d', d => (d.group === 0 && d.label !== 'Origin') ? '' : shape(d.group))
    .attr('stroke', 'black')
    .attr('stroke-width', '1px')
    .attr('stroke-opacity', '0.75')


  // Add the X Axis
  let x_axis_g = svg.append('g')
    .attr('transform', 'translate(0,' + height + ')')
    .classed('axis', true)
    .call(d3.axisBottom(x));

  // Add the Y Axis
  let y_axis_g = svg.append('g')
    .classed('axis', true)
    .call(d3.axisLeft(y));

  // Draw the bias direction arrow
  let arrow_endpoints = data.filter(d => d.group === 0).map(d => [x(d.x), y(d.y)]);

  let bias_line = svg.append('path')
    .attr('id', 'bias-direction-line')
    .attr('d', d3.line()(arrow_endpoints))
    .attr('stroke', '#5b5b5b')
    .attr('stroke-width', '4px');

  let zoom = d3.zoom().scaleExtent([0.5, 20]).extent([[0, 0], [width, height]]).on("zoom", update_plot);

  parent_svg.append('rect')
    .attr('width', width)
    .attr('height', height)
    .attr('fill', 'none')
    .attr('pointer-events', 'all')
    // .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
    .lower();

  parent_svg.call(zoom)

  function update_plot() {
    let newX = d3.event.transform.rescaleX(x);
    let newY = d3.event.transform.rescaleY(y);

    // update axes with these new boundaries
    x_axis_g.call(d3.axisBottom(newX))
    y_axis_g.call(d3.axisLeft(newY));

    datapoint_group.transition()
      .duration(0)
      .attr('transform', d => 'translate(' + newX(d.x) + ',' + newY(d.y) + ')');

    //
    bias_line.attr('d', d3.line()(data.filter(d => d.group === 0).map(d => [newX(d.x), newY(d.y)])));
  }
}

function draw_scatter_anim(svg, point_data, neighbor_data, x, y, width, height, margin) {
  // Add the scatterplot
  svg.append('defs')
    .append('marker')
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
    .on('click', function (d) {
      let neigbors_base = neighbor_data.base[d.label]
      let neighbors_debiased = neighbor_data.debiased[d.label];

      d3.select('#knn').select('#base-neighbors')
        .selectAll('span')
        .data(neigbors_base)
        .join('span')
        .classed('neighbor-item', true)
        .html(d => d);

      d3.select('#knn').select('#debiased-neighbors')
        .selectAll('span')
        .data(neighbors_debiased)
        .join('span')
        .classed('neighbor-item', true)
        .html(d => d);
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

  // Remove buttons
  datapoint_group.append('text')
    .attr('class', 'fa cross-button')
    .attr('x', 10)
    .attr('y', -10)
    .attr('visibility', 'hidden')
    .on('click', remove_point)
    .text('\uf057');

  datapoint_group.append('path')
    .attr('fill', d => d.group === 0 ? '#414141' : color(d.group))
    .attr('d', d => (d.group === 0 && d.label != 'Origin') ? '' : shape(d.group))
    .attr('stroke', 'black')
    .attr('stroke-width', '1px')
    .attr('stroke-opacity', '0.75')

  // Draw the bias direction arrow
  let arrow_endpoints = [point_data.filter(d => d.group === 0).map(d => [d.x, d.y])];

  if ($('#algorithm-selection-button').text() === 'Algorithm: OSCaR') {
    let line_data = [[arrow_endpoints[0][0], arrow_endpoints[0][1]], [arrow_endpoints[0][2], arrow_endpoints[0][3]]];
    console.log(line_data);

    let bias_lines = svg.append('g')
      .attr('id', 'bias-direction-group')
      .selectAll('path')
      .data(line_data)
      .join('path')
      .attr('d', d => d3.line()(d.map(d => [x(d[0]), y(d[1])])))
      .attr('id', (d, i) => 'bias-direction-line' + i)
      .attr('class', 'bias-line')
      .attr('stroke', '#5b5b5b')
      .attr('stroke-width', '4px');

    d3.select('#bias-direction-group').selectAll('text')
      .data(line_data)
      .join('text')
      .attr('class', 'group-0')
      .attr('dy', -10)
      .style('font-size', '1.2em')
      .attr('id', (d, i) => 'text' + i)
      .append('textPath')
      .attr('xlink:href', (d, i) => '#bias-direction-line' + i)
      .attr('startOffset', '30')
      .text((d, i) => point_data.filter(d => d.group === 0 && d.label !== 'Origin').map(d => d.label)[i] + '➞');
  } else {
    let bias_line = svg.append('path')
      .data(arrow_endpoints)
      .attr('id', 'bias-direction-line')
      .attr('d', d => d3.line()(d.map(d => [x(d[0]), y(d[1])])))
      .attr('stroke', '#5b5b5b')
      .attr('stroke-width', '4px');

    svg.append('text')
      .attr('class', 'group-0')
      .attr('dy', -10)
      .style('font-size', '1.2em')
      .append('textPath')
      .attr('xlink:href', '#bias-direction-line')
      .attr('startOffset', '30')
      .text(point_data.filter(d => d.group === 0 && d.label !== 'Origin').map(d => d.label) + '➞');
  }

  zoom = d3.zoom().scaleExtent([0.02, 20]).extent([[0, 0], [width, height]]).on("zoom", update_plot);

  svg.append('rect')
    .attr('width', width)
    .attr('height', height)
    .attr('fill', 'none')
    // .attr('stroke', 'black')
    // .attr('rx', 15)
    .attr('pointer-events', 'all')
    .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
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

    d3.select('path#bias-direction-line').attr('d', d => d3.line()(d.map(d => [newX(d[0]), newY(d[1])])));
    d3.selectAll('.bias-line').attr('d', d => d3.line()(d.map(d => [newX(d[0]), newY(d[1])])));
  }
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

function add_groups(data) {
  let grouped_data = [];

  for (let i = 0; i < data['vectors1'].length; i++) {
    grouped_data.push({'position': data['vectors1'][i], 'label': data['words1'][i], 'group': 1});
  }

  for (let i = 0; i < data['vectors2'].length; i++) {
    grouped_data.push({'position': data['vectors2'][i], 'label': data['words2'][i], 'group': 2});
  }

  for (let i = 0; i < data['evalvecs'].length; i++) {
    grouped_data.push({'position': data['evalvecs'][i], 'label': data['evalwords'][i], 'group': 3});
  }

  return grouped_data;
}

function compute_axes_limits_sym(points) {
  points_without_subspace = points.filter(x => x.group !== 0);
  let x_coords = points_without_subspace.map(d => Math.abs(d.x));
  let y_coords = points_without_subspace.map(d => Math.abs(d.y));
  let x = Math.max(...x_coords), y = Math.max(...y_coords);
  coord_max = Math.max(x, y);
  return {
    // x_min: -x - 0.2 * x, x_max: x + 0.2 * x,
    // y_min: -y - 0.2 * y, y_max: y + 0.2 * y
    // x_min: -1.1, x_max: 1.1, y_min: -1.1, y_max: 1.1
    // x_min: -1.1, x_max: 1.1, y_min: -1.1, y_max: 1.1
    x_min: -coord_max - 0.2 * coord_max, x_max: coord_max + 0.2 * coord_max,
    y_min: -coord_max - 0.2 * coord_max, y_max: coord_max + 0.2 * coord_max
  }
}

function compute_axes_limits(points) {
  let x_coords = points.map(d => d.x);
  let x_min = Math.min(...x_coords), x_max = Math.max(...x_coords);
  let y_coords = points.map(d => d.y);
  let y_min = Math.min(...y_coords), y_max = Math.max(...y_coords);
  return {
    x_min: x_min - 0.2 * Math.abs(x_min), x_max: x_max + 0.2 * Math.abs(x_max),
    y_min: y_min - 0.2 * Math.abs(y_min), y_max: y_max + 0.2 * Math.abs(y_max)
  }
}

function compute_perpendicular(line) {
  let x = line[0], y = line[1];
  if (Math.abs(x) <= 0.0000001) {
    return [0, 0]
  }
  return [-line[1] / line[0], 1];
}

function setup_animation(anim_svg, response, identifier) {
  try {
    console.log(response);

    function update_anim_svg(svg, x_axis, y_axis, step, transition_info, camera_step = false) {
      let explanation_text = step <= response.explanations.length ? response.explanations[step] : 'No explanation found.';
      $('#explanation-text').text(explanation_text);

      svg.attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

      // Adjust the axis
      let axes_limits = compute_axes_limits_sym(response.anim_steps[step]);
      let x_axis_obj = svg.select('.x');
      let y_axis_obj = svg.select('.y');
      x_axis.domain([axes_limits['x_min'], axes_limits['x_max']]).nice();
      y_axis.domain([axes_limits['y_min'], axes_limits['y_max']]).nice();

      if (camera_step) {
        x_axis_obj.transition().duration(ANIMATION_DURATION).ease(INTERPOLATION)
          .on('start', function () {
            d3.select('#camera-indicator').classed('animate-flicker', true).attr('visibility', 'visible');
          })
          .on('end', function () {
            d3.select('#camera-indicator').classed('animate-flicker', false).attr('visibility', 'hidden');
          })
          .call(d3.axisBottom(x_axis))
        y_axis_obj.transition().duration(ANIMATION_DURATION).ease(INTERPOLATION).call(d3.axisLeft(y_axis));
      } else {
        x_axis_obj.transition().duration(ANIMATION_DURATION).ease(INTERPOLATION).call(d3.axisBottom(x_axis));
        y_axis_obj.transition().duration(ANIMATION_DURATION).ease(INTERPOLATION).call(d3.axisLeft(y_axis));
      }

      labelArray = [];
      anchorArray = [];

      for (let i = 0; i < response.anim_steps[step].length; i++) {
        let d = response.anim_steps[step][i];
        let x_coord = x_axis(d.x), y_coord = y_axis(d.y)
        labelArray.push({x: x_coord, y: y_coord, name: d.label})
        anchorArray.push({x: x_coord, y: y_coord, name: d.label, r: 10})
      }

      if (DYNAMIC_PROJ && transition_info.index !== -1) {
        let transition_array = transition_info.forward ? response.transitions[transition_info.index] : response.transitions[transition_info.index].slice().reverse();
        console.log(transition_array)
        let i = 0;
        svg.selectAll('g')
          .data(transition_array[i])
          .transition()
          .duration(ANIMATION_DURATION)
          .ease(INTERPOLATION)
          .attr('transform', d => 'translate(' + x_axis(d.x) + ',' + y_axis(d.y) + ')')
          .end()
          .then(function repeat() {
            i += 1;
            if (i < transition_array.length) {
              {
                svg.selectAll('g')
                  .data(transition_array[i])
                  .transition()
                  .duration(ANIMATION_DURATION)
                  .ease(INTERPOLATION)
                  .attr('transform', d => 'translate(' + x_axis(d.x) + ',' + y_axis(d.y) + ')')
                  .end()
                  .then(repeat)
              }
            }
          })
      } else {
        let datapoint_group = svg.selectAll('g')
          .data(response.anim_steps[step])
          .transition()
          .duration(ANIMATION_DURATION)
          .ease(INTERPOLATION)
          .on('start', () => {
            d3.select('g#animationgroup').on('.zoom', null)
          })
          .on('end', () => {
            d3.select('g#animationgroup').call(zoom.transform, d3.zoomIdentity);
            d3.select('g#animationgroup').call(zoom)
          })
          .attr('transform', d => {
            return 'translate(' + x_axis(d.x) + ',' + y_axis(d.y) + ')'
          })

        // Draw labels
        let labels = datapoint_group.selectAll('text.class-label')

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
          .attr('x', d => filterByName(d.label, labelArray, 'name').x - filterByName(d.label, anchorArray, 'name').x)
          .attr('y', d => filterByName(d.label, labelArray, 'name').y - filterByName(d.label, anchorArray, 'name').y)
      }

      let arrow_endpoints = [response.anim_steps[step].filter(d => d.group === 0).map(d => [d.x, d.y])];

      if ($('#algorithm-selection-button').text() === 'Algorithm: Iterative Null Space Projection') {
        let classifier_line = response.anim_steps[step].filter(d => d.group === 0).map(d => [d.x, d.y]);
        classifier_line[1] = compute_perpendicular(classifier_line[1]);
        classifier_line[0] = classifier_line[1].map(d => -d);
        classifier_line = classifier_line.map(d => [x_axis(d[0] * 10), y_axis(d[1] * 10)]);

        svg.select('#classification-line')
          .transition()
          .ease(INTERPOLATION)
          .duration(ANIMATION_DURATION)
          .attr('d', d3.line()(classifier_line));
      } else {
        svg.select('#classification-line')
          .selectAll('*')
          .remove();
      }

      if ($('#algorithm-selection-button').text() === 'Algorithm: OSCaR') {
        let line_data = [[arrow_endpoints[0][0], arrow_endpoints[0][1]], [arrow_endpoints[0][2], arrow_endpoints[0][3]]];

        svg.select('#bias-direction-group')
          .selectAll('path')
          .data(line_data)
          .transition()
          .ease(INTERPOLATION)
          .duration(ANIMATION_DURATION)
          .attr('d', d => d3.line()(d.map(d => [x_axis(d[0]), y_axis(d[1])])));
      } else {
        svg.select('#bias-direction-line')
          .data(arrow_endpoints)
          .transition()
          .ease(INTERPOLATION)
          .duration(ANIMATION_DURATION)
          .attr('d', d => d3.line()(d.map(d => [x_axis(d[0]), y_axis(d[1])])));
      }
    }

    let margin = {top: 20, right: 20, bottom: 20, left: 40};
    let width = anim_svg.node().width.baseVal.value - margin.left - margin.right;
    let height = anim_svg.node().height.baseVal.value - margin.top - margin.bottom;

    // set the ranges
    let x_axis = d3.scaleLinear().range([0, width - 30]);
    let y_axis = d3.scaleLinear().range([height, 0]);
    let axes_limits = compute_axes_limits_sym(response.anim_steps[0]);
    x_axis.domain([axes_limits['x_min'], axes_limits['x_max']]).nice();
    y_axis.domain([axes_limits['y_min'], axes_limits['y_max']]).nice();

    let camera_icon = anim_svg.append('image')
      .attr('id', 'camera-indicator')
      .attr('x', 50)
      .attr('y', height - 30)
      .attr('href', 'static/assets/camera.svg')
      .attr('visibility', 'hidden')

    let step_indicator = anim_svg.append('text')
      .attr('id', 'step-indicator')
      .attr('x', width)
      .attr('y', 25)
      .attr('text-anchor', 'end')
      .text('Step=0');


    let svg = anim_svg.append('g')
      .attr('id', identifier + 'group')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    draw_scatter_anim(svg, response.anim_steps[0], response.knn, x_axis, y_axis, width, height, margin);
    let axes = draw_axes(svg, width, height, x_axis, y_axis);

    svg.append('path')
      .attr('id', 'classification-line')
      .attr('stroke', '#2751ac')
      .attr('d', d3.line()([[0, 0], [1, 1]]))
      .attr('stroke-width', '2px')
      .attr('stroke-dasharray', '5, 5')

    let x_axes_obj = axes[0], y_axes_obj = axes[1];
    $('#explanation-text').text(response.explanations[0]);

    // Step back
    let step_backward_btn = $('#play-control-sb');
    let step_forward_btn = $('#play-control-sf');
    let fast_backward_btn = $('#play-control-fb');
    let fast_forward_btn = $('#play-control-ff');

    btn_active(step_backward_btn, false);
    btn_active(fast_backward_btn, false);
    btn_active(step_forward_btn, true);
    btn_active(fast_forward_btn, true);

    // Step backward
    step_backward_btn.unbind('click').on('click', function (e) {
      // if already at 0, do nothing
      if (ANIMSTEP_COUNTER === 0) {
        console.log('Already at first step');
      } else {
        ANIMSTEP_COUNTER -= 1;
        btn_active(step_forward_btn, true);
        btn_active(fast_forward_btn, true);

        update_anim_svg(svg, x_axis, y_axis, ANIMSTEP_COUNTER, {
          index: ANIMSTEP_COUNTER,
          forward: false
        }, response.camera_steps[ANIMSTEP_COUNTER + 1]);

        if (ANIMSTEP_COUNTER === 0) {
          btn_active(step_backward_btn, false);
          btn_active(fast_backward_btn, false);
        }
      }
      d3.select('#step-indicator').text(`Step=${ANIMSTEP_COUNTER}`);
    })

    // To the first step
    fast_backward_btn.unbind('click').on('click', function (e) {
      if (ANIMSTEP_COUNTER === 0) {
        console.log('Already at first step');
      } else {
        ANIMSTEP_COUNTER = 0;
        btn_active(step_forward_btn, true);
        btn_active(fast_forward_btn, true);

        update_anim_svg(svg, x_axis, y_axis, ANIMSTEP_COUNTER, {index: -1, forward: false}, true);

        if (ANIMSTEP_COUNTER === 0) {
          btn_active(step_backward_btn, false);
          btn_active(fast_backward_btn, false);
        }
      }
      d3.select('#step-indicator').text(`Step=${ANIMSTEP_COUNTER}`);

    })

    // Step forward
    step_forward_btn.unbind('click').on('click', function (e) {
      if (ANIMSTEP_COUNTER === response.anim_steps.length - 1) {
        console.log('Already at last step');
      } else {
        ANIMSTEP_COUNTER += 1;
        btn_active(step_backward_btn, true);
        btn_active(fast_backward_btn, true);

        update_anim_svg(svg, x_axis, y_axis, ANIMSTEP_COUNTER, {index: ANIMSTEP_COUNTER - 1, forward: true}, response.camera_steps[ANIMSTEP_COUNTER]);

        if (ANIMSTEP_COUNTER === response.anim_steps.length - 1) {
          btn_active(step_forward_btn, false);
          btn_active(fast_forward_btn, false);
        }
      }
      d3.select('#step-indicator').text(`Step=${ANIMSTEP_COUNTER}`);
    })

    // To the last step
    fast_forward_btn.unbind('click').on('click', function (e) {
      if (ANIMSTEP_COUNTER === response.anim_steps.length - 1) {
        console.log('Already at last step');
      } else {
        ANIMSTEP_COUNTER = response.anim_steps.length - 1;
        btn_active(step_backward_btn, true);
        btn_active(fast_backward_btn, true);

        update_anim_svg(svg, x_axis, y_axis, ANIMSTEP_COUNTER, {index: -1, forward: true}, true);

        if (ANIMSTEP_COUNTER === response.anim_steps.length - 1) {
          btn_active(step_forward_btn, false);
          btn_active(fast_forward_btn, false);
        }
      }
      d3.select('#step-indicator').text(`Step=${ANIMSTEP_COUNTER}`);

    })
  } catch (e) {
    console.log(e);
  }
}

function btn_active(btn, bool_active) {
  btn.prop('disabled', !bool_active);
}

function captureEnter(e) {
  if (e.key === 'Enter' || e.keyCode === 13) {
    $(e.target).blur();
    $('#seedword-form-submit').click();
    $('#example-selection-button').html('Choose an example or provide seedword sets below')
  }
}

// Functionality for the dropdown-menus
$('#example-dropdown a').click(function (e) {
  $('#example-selection-button').text(this.innerHTML);
});

$('#algorithm-dropdown a').click(function (e) {
  $('#example-selection-button').html('Choose an example or provide seedword sets below');

  let algorithm = this.innerHTML;
  let subspace_selector = $('#subspace-dropdown-items').children();

  $('#algorithm-selection-button').text('Algorithm: ' + algorithm);
  subspace_selector.removeClass('disabled');

  if (algorithm === 'Linear projection') {
    subspace_selector.addClass('disabled');
    subspace_selector[1].click();
    $(subspace_selector[1]).removeClass('disabled');
    $(subspace_selector[2]).removeClass('disabled');
    $(subspace_selector[3]).removeClass('disabled');
    $(subspace_selector[5]).removeClass('disabled');
  }

  if (algorithm === 'Hard debiasing') {
    $('#equalize-holder').show();
    subspace_selector.addClass('disabled');
    subspace_selector[1].click();
    $(subspace_selector[1]).removeClass('disabled');
    $(subspace_selector[2]).removeClass('disabled');
    $(subspace_selector[3]).removeClass('disabled');
    $(subspace_selector[5]).removeClass('disabled');
  } else {
    $('#equalize-holder').hide();
  }

  if (algorithm === 'OSCaR') {
    $('#input-two-col-oscar').show();
    subspace_selector.addClass('disabled');
    subspace_selector[1].click();
    $(subspace_selector[1]).removeClass('disabled');
    $(subspace_selector[2]).removeClass('disabled');
    $(subspace_selector[3]).removeClass('disabled');
    $(subspace_selector[5]).removeClass('disabled');
  } else {
    $('#input-two-col-oscar').hide();
  }
  if (algorithm === 'Iterative Null Space Projection') {
    subspace_selector.addClass('disabled');
    subspace_selector[4].click();
    $(subspace_selector[4]).removeClass('disabled');
  }
});

$('#subspace-dropdown a').click(function (e) {
  try {
    $('#example-selection-button').html('Choose an example or provide seedword sets below')

    let subspace_method = this.innerHTML;
    $('#subspace-selection-button').text('Subspace method: ' + subspace_method);

    if (subspace_method === 'PCA') {
      // let seedwords1 = $('#seedword-text-1').val()
      // let seedwords2 = $('#seedword-text-2').val();
      // $('#seedword-text-1').val(seedwords1 + ', ' + seedwords2)
      $('#seedword-text-2').hide();
    } else if (subspace_method === 'PCA-paired') {
      $('#seedword-text-2').hide();
    } else if (subspace_method === 'Two means' || subspace_method === 'Classification' || subspace_method === 'GSS') {
      $('#seedword-text-2').show();
    } else {
      console.log('Incorrect subspace method');
    }
  } catch (e) {
    console.log(e);
  }
});

// Functionality for various toggle buttons
$('#data-label-chk').click(function () {
  if (LABEL_VISIBILITY === true) {
    d3.selectAll('.class-label').attr('hidden', true);
    d3.select('#toggle-label-icon').attr('class', 'fa fa-toggle-on fa-rotate-180');
  } else {
    d3.selectAll('.class-label').attr('hidden', null);
    d3.select('#toggle-label-icon').attr('class', 'fa fa-toggle-on');
  }
  LABEL_VISIBILITY = !LABEL_VISIBILITY;
});

$('#toggle-eval-chk').click(function () {

  let eval_state = $(this).prop('checked');
  if (eval_state === false) {
    d3.selectAll('.group-3').attr('hidden', true);
  } else {
    d3.selectAll('.group-3').attr('hidden', null);
  }
});

$('#toggle-mean-chk').click(function () {
  if (MEAN_VISIBILITY === true) {
    d3.selectAll('#bias-direction-line').attr('hidden', true);
    d3.selectAll('.group-0').attr('hidden', true);
    d3.select('#toggle-mean-icon').attr('class', 'fa fa-toggle-on fa-rotate-180');
  } else {
    d3.selectAll('#bias-direction-line').attr('hidden', null);
    d3.selectAll('.group-0').attr('hidden', null);
    d3.selectAll('#toggle-mean-icon').attr('class', 'fa fa-toggle-on');
  }
  MEAN_VISIBILITY = !MEAN_VISIBILITY;
});

$('#remove-points-chk').click(function () {
  let cross_buttons = d3.selectAll('.cross-button');
  let chk_state = $(this).prop('checked');
  if (chk_state === false) {
    cross_buttons.attr('visibility', 'hidden');
  } else {
    cross_buttons.filter(d => d.group !== 0).attr('visibility', 'visible');
    // .classed('shaker', !cross_buttons.classed('shaker'));
  }
});

function svg_cleanup() {
  $('#pre-debiased-svg').empty();
  $('#animation-svg').empty();
  $('#post-debiased-svg').empty();
}

// Functionality for the 'Run' button
$('#seedword-form-submit').click(function () {
  try { // Perform cleanup
    svg_cleanup();
    ANIMSTEP_COUNTER = 0;

    $('#toggle-eval-chk').prop('checked', true);
    $('#toggle-mean-chk').prop('checked', true);
    $('#data-label-chk').prop('checked', true);
    if ($('#remove-points-chk').prop('checked', true)) {
      $('#remove-points-chk').click()
    }

    let seedwords1 = $('#seedword-text-1').val();
    let seedwords2 = $('#seedword-text-2').val();
    let evalwords = $('#evaluation-list').val();
    let equalize = $('#equalize-list').val()
    let orth_subspace = $('#oscar-seedword-text-1').val();
    let algorithm = $('#algorithm-selection-button').text();
    let subspace_method = $('#subspace-selection-button').text();
    let concept1_name = $('#concept-label-1').val();
    let concept2_name = $('#concept-label-2').val();

    $.ajax({
      type: 'POST',
      url: '/seedwords2',
      data: {
        seedwords1: seedwords1,
        seedwords2: seedwords2,
        evalwords: evalwords,
        equalize: equalize,
        orth_subspace: orth_subspace,
        algorithm: algorithm,
        subspace_method: subspace_method,
        concept1_name: concept1_name,
        concept2_name: concept2_name
      },
      beforeSend: function () {
        $('.overlay').addClass('d-flex').show();
        $('#spinner-holder').show();
        $('#seedword-form-submit').attr('disabled', 'disabled');
        $('#weat-display').text('')
      },
      success: function (response) {
        // let predebiased_svg = d3.select('#pre-debiased-svg');
        // draw_scatter_static(predebiased_svg, response, 'Pre-debiasing', false,);

        let animation_svg = d3.select('#animation-svg');
        // draw_svg_scatter(animation_svg, response, 'Pre-debiasing', true, true);
        setup_animation(animation_svg, response, 'animation')

        // let postdebiased_svg = d3.select('#post-debiased-svg');
        // draw_scatter_static(postdebiased_svg, response, 'Post-debiasing', true,);

        // $('#weat-predebiased').html('WEAT score = ' + response['weat_scores']['pre-weat'].toFixed(3));
        // $('#weat-postdebiased').html('WEAT score = ' + response['weat_scores']['post-weat'].toFixed(3));
        // console.log(response.weat_scores);

        // enable toolbar buttons
        d3.select('#toggle-labels-btn').attr('disabled', null);
        d3.select('#remove-points-chk').attr('disabled', null);
        d3.select('#toggle-mean-btn').attr('disabled', null);
        d3.select('#toggle-eval-btn').attr('disabled', null);

        if (LABEL_VISIBILITY === false) {
          LABEL_VISIBILITY = true;
          $('#toggle-labels-btn').click();
        }
      },
      complete: function () {
        $('.overlay').removeClass('d-flex').hide();
        $('#spinner-holder').hide();
        $('#seedword-form-submit').removeAttr('disabled');
      },
      error: function (request, status, error) {
        alert(request.responseJSON.message);
      }
    });
  } catch (e) {
    console.log(e);
  }
});

$('#freeze-embedding').click(function () {
  $.ajax({
    type: 'POST',
    url: '/freeze',
    success: function (response) {
      // alert('Successfully set current debiased embedding as the new base embedding.');
      $('#freeze-embedding').prop('disabled', true)
      $('#unfreeze-embedding').prop('disabled', false)
    },
    error: function (response) {
      alert('Something went wrong');
    }
  });
});

$('#unfreeze-embedding').click(function () {
  $.ajax({
    type: 'POST',
    url: '/unfreeze',
    success: function (response) {
      // alert('Successfully set current debiased embedding as the new base embedding.');
      $('#freeze-embedding').prop('disabled', false)
      $('#unfreeze-embedding').prop('disabled', true)
    },
    error: function (response) {
      alert('Something went wrong');
    }
  });
});

// Allow enter in text inputs to press Run button
$('#seedword-text-1').on('keyup', captureEnter);
$('#seedword-text-2').on('keyup', captureEnter);
$('#evaluation-list').on('keyup', captureEnter);
$('#equalize-list').on('keyup', captureEnter);
$('#oscar-seedword-text-1').on('keyup', captureEnter);

// Preloaded examples
$('#preloaded-examples').on('click', function () {
  $("#example-dropdown").empty();
  $.getJSON('static/assets/examples.json', {ts: new Date().getTime()}, function (examples) {
    examples.data.forEach(function (example, index) {
      let dropdown = d3.select('#example-dropdown');
      let dropdown_item = dropdown.append('a')
        .classed('dropdown-item', true)
        .classed(index === 0 ? 'active' : '', true)
        .text((index + 1) + '. ' + example.name);
      dropdown_item.on('click', function () {
        $('#algorithm-dropdown').children()[ALGO_MAP[example.algorithm]].click();
        $('#subspace-dropdown-items').children()[SUBSPACE_MAP[example.subspace]].click();

        $('#example-selection-button').text('Chosen example: ' + (index + 1) + '. ' + example.name);

        $('#concept-label-1').val('Concept1');
        $('#concept-label-2').val('Concept2');

        if (example.hasOwnProperty('seedwords-1')) {
          $('#seedword-text-1').val(example["seedwords-1"]);
        }
        if (example.hasOwnProperty('seedwords-2')) {
          $('#seedword-text-2').val(example["seedwords-2"]);
        }
        if (example.hasOwnProperty('equalize')) {
          $('#equalize-list').val(example["equalize"]);
        }
        if (example.hasOwnProperty('evalset')) {
          $('#evaluation-list').val(example["evalset"]);
        }
        if (example.hasOwnProperty('oscar-c2-seedwords')) {
          $('#oscar-seedword-text-1').val(example["oscar-c2-seedwords"]);
        }
        if (example.hasOwnProperty('concept1')) {
          $('#concept-label-1').val(example['concept1']);
        }
        if (example.hasOwnProperty('concept2')) {
          $('#concept-label-2').val(example['concept2']);
        }
        $('#seedword-form-submit').click();
      })
    })
  }).fail(function (e) {
    console.log(e);
  })
})

$('#control-collapser').on('click', function () {
  console.log(this.innerHTML)
  if (this.innerHTML === 'Hide controls') {
    this.innerHTML = 'Show controls';
  } else if (this.innerHTML === 'Show controls') {
    this.innerHTML = 'Hide controls';
  }
})

$('#import-btn').on('click', function () {
  $('#import-input').click();
})

$('#weat-btn').on('click', function () {
  $.ajax({
    'url': '/weat',
    'success': function(response) {
      console.log(response);
      $('#weat-display').text(`WEAT: ${response.weat_scores['pre-weat'].toFixed(3)} \u21E8 ${response.weat_scores['post-weat'].toFixed(3)}`)
    },
    'error': function(response) {
      console.log(response);
    }
  })
});

if (TESTING) {
  try { // $('#seedword-text-1').val('mike, lewis, noah, james, lucas, william, jacob, daniel, henry, matthew');
    // $('#seedword-text-2').val('lisa, emma, sophia, emily, chloe, hannah, lily, claire, anna');
    // $('#seedword-text-1').val('WATERDOGTAVERN1');
    // $('#seedword-text-2').val('LOSANGELESAIRPORT');
    // $('#evaluation-list').val('CSMCCAFETERIA, UNIVOFSOUTHERNCAL, SUNRICECAFE, EPICUREANATNOTREDAME');
    // $('#equalize-list').val('monastery-convent, spokesman-spokeswoman, dad-mom, men-women, councilman-councilwoman,' +
    //   ' grandpa-grandma, grandsons-granddaughters, testosterone-estrogen, uncle-aunt, wives-husbands, father-mother,' +
    //   ' grandpa-grandma, he-she, boy-girl, boys-girls, brother-sister, brothers-sisters, businessman-businesswoman,' +
    //   ' chairman-chairwoman, colt-filly, congressman-congresswoman, dads-moms, dudes-gals, father-mother, fatherhood-motherhood,' +
    //   ' fathers-mothers, fella-granny, fraternity-sorority, gelding-mare, gentleman-lady, gentlemen-ladies,' +
    //   ' grandfather-grandmother, grandson-granddaughter, he-she, himself-herself, his-her, king-queen, kings-queens,' +
    //   ' male-female, males-females, man-woman, men-women, nephew-niece, prince-princess, schoolboy-schoolgirl, son-daughter, sons-daughters')
    // $('#oscar-seedword-text-1').val('scientist, doctor, nurse, secretary, maid, dancer, cleaner, advocate, player, banker')
    // $('#algorithm-dropdown').children()[1].click();
    // $('#subspace-dropdown-items').children()[1].click();
    // $('#seedword-form-submit').click();
    $('#example-selection-button').click();
    setTimeout(() => {
      $('#example-dropdown').children()[0].click();
    }, 600)
  } catch (e) {
    console.log(e);
  }
}
