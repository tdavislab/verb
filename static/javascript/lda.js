let color = d3.scaleOrdinal(d3.schemeCategory10);

function preprocess(data) {
  let labels = [];
  let datasets = []
  let scatterData = {}
  data.forEach((datapoint) => {
    let processed_point = {x: parseFloat(datapoint.x), y: parseFloat(datapoint.y), label: datapoint.words}
    if (datapoint.pos in scatterData) {
      scatterData[datapoint.pos].push(processed_point)
    } else {
      scatterData[datapoint.pos] = [];
    }
  });

  let chartColors = [
    'rgb(255, 99, 132)',
    'rgb(255, 159, 64)',
    'rgb(255, 205, 86)',
    'rgb(69,177,122)',
    'rgb(54, 162, 235)',
    'rgb(153, 102, 255)',
    'rgb(122,108,21)',
    'rgb(0,53,146)',
    'rgb(143,58,35)',
    'rgb(146,21,157)',
    'rgb(75,121,121)',
    'rgb(10,103,27)',
  ];

  let scatterDataArr = Object.entries(scatterData);
  scatterDataArr.forEach(([pos, datapoints], idx) => {
      datasets.push({
        label: pos,
        data: datapoints,
        // backgroundColor: chartColors[idx],
        backgroundColor: color(pos),
      })
    }
  )

  return {datasets}
}

class Projection {
  constructor(canvas) {
    this.canvas = canvas;
  }

  setData(data) {
    try {
      this.data = preprocess(data);
      this.config = {
        type: 'scatter',
        data: this.data,
        options: {
          responsive: false,
          scales: {
            x: {min: -1.2, max: 1.2},
            y: {min: -1.2, max: 1.2}
          },
          animation: {
            duration: 0
          },
          plugins: {
            legend: {
              position: 'right'
            },
            tooltip: {
              callbacks: {
                label: x => '',
                footer: function (tooltipItems) {
                  return tooltipItems.map(d => d.raw.label)
                }
              }
            }
          }
        }
      }
      this.chart = new Chart(this.canvas, this.config);
      console.log(this.chart)
    } catch (e) {
      console.log(e);
    }
  }

  dimDataset(label) {

  }
}

function draw() {
  d3.csv('data/lda5k.csv')
    .then(data => {
      let projection = new Projection(document.getElementById('projection'));
      projection.setData(data);
      draw_treemap(data)
    })
    .catch(error => {
      console.log(error);
    })
}

function draw_treemap(data) {
  let svg = d3.select('#treemap');
  let labelwiseData = {};
  data.forEach((datapoint) => {
    let processed_point = {x: parseFloat(datapoint.x), y: parseFloat(datapoint.y), label: datapoint.words}
    if (datapoint.pos in labelwiseData) {
      labelwiseData[datapoint.pos].push(processed_point)
    } else {
      labelwiseData[datapoint.pos] = [];
    }
  });

  console.log(labelwiseData)
  let h_data = {
    name: 'root',
    children: Object.entries(labelwiseData).map(x => {return {name: x[0] + ` (${x[1].length})`, value: x[1].length}})
  }


  let treemap = d3.treemap().size([800, 800]).padding(1)
  let root = treemap(d3.hierarchy(h_data).sum(d => d.value).sort((a, b) => b.value - a.value));

  let leaf = svg.selectAll('g')
    .data(root.leaves())
    .join('g')
    .attr("transform", d => `translate(${d.x0},${d.y0})`);

  leaf.append('rect')
    .attr("fill", d => {
      while (d.depth > 1) d = d.parent;
      return color(d.data.name);
    })
    .attr("fill-opacity", 0.6)
    .attr("width", d => d.x1 - d.x0)
    .attr("height", d => d.y1 - d.y0);

  leaf.append('title')
    .text(d => d.data.name)

  leaf.append('text')
    .attr('x', 5)
    .attr('y', 20)
    .text(d => d.data.name)
}

draw()
