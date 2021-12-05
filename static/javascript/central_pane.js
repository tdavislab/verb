function createParallelCoord(data) {
  // linear color scale
  $("#parallel_coord").empty();

  var pc = d3
    .parcoords()("#parallel_coord")
    .data(data)
    .bundlingStrength(0) // set bundling strength
    .smoothness(0)
    //.bundleDimension("gender")
    .bundleDimension(Object.keys(data[0])[1])
    .hideAxis(["word"])
    .composite("darken")
    .color("steelblue") // quantitative color scale
    .alpha(0.6)
    .mode("queue")
    //.render()
    .brushMode("1D-axes-multi") // enable brushing
    .interactive() // command line mode
    .reorderable();

  return pc;
}

//________________________________________________________FROM HERE_________//
// This code was originally in  left_pane.js
// Given a list as input, populate drop down menu with each element as an option
function populateDropDownList(data) {
  var option = "";
  for (var i = 0; i < data.length; i++) {
    if (data[i] != "word")
      option += '<option value="' + data[i] + '">' + data[i] + "</option>";
  }
  return option;
}
function changeTarget(selVal) {
  // path = './data/wordList/target/'+selVal
  if (current_embedding == "Hindi fastText") {
    path = "./data/wordList/target/hi/" + selVal;
  } else if (current_embedding == "French fastText") {
    path = "./data/wordList/target/fr/" + selVal;
  } else {
    path = "./data/wordList/target/en/" + selVal;
  }
  console.log("selval is: ", selVal);
  if (selVal == "Custom") {
    $("#target").val("");
    return;
  }
  $.get(
    "/getWords",
    {
      path: path,
    },
    (res) => {
      console.log(res);
      $("#target").val(res["target"].join());
    }
  );
}

$("#highlight_words").click(function () {
  console.log("Highlight button clicked");
  // afterHighlight =  true
  inSearch = true;
  filter_words = [];
  text = $("#target").val().toLowerCase();
  // Regex expression to split by newline and comma
  // https://stackoverflow.com/questions/34316090/split-string-on-newline-and-comma
  // https://stackoverflow.com/questions/10346722/how-can-i-split-a-javascript-string-by-white-space-or-comma
  text = text.split(/[\n, ]+/);
  for (i = 0; i < text.length; i++) {
    if (text[i].length > 0) {
      filter_words.push(text[i]);
    }
  }
  // updateProgressBar(filter_words);
  highlightWords(null, (neighbors = filter_words));
});

//________________________________________________________UPTIL HERE_________//

/*
Top pane events
Parallel coordinates properties
smoothness, alpha. etc.
paracoord.js library events
*/
$("#alpha_input").on("change", function () {
  alpha = parseFloat($(this).val());
  pc.alpha(alpha).render();
  $("#alpha_text").html(alpha);
});

$("#smoothness_input").on("change", function () {
  smooth = parseFloat($(this).val());
  pc.smoothness(smooth).render();
  $("#smoothness_text").html(smooth);
});

$("#bundle_input").on("change", function () {
  bundle = +$(this).val();
  pc.bundlingStrength(bundle).render();
  $("#bundle_text").html(bundle);
});

$("#bundle_dimension").on("change", function () {
  value = $(this).val();
  console.log(value);
  pc.bundleDimension(value);
});

// Reset brush button -- removes all brushes
$("#reset_brush").on("click", brush_reset);

function brush_reset() {
  pc.brushReset();
  d3.selectAll(".extentLabels").remove();

  if (inSearch) {
    populate_neighbors(highlighted_data);
    updateProgressBar(highlighted_data);

    d3.selectAll([pc.canvas["highlight"]]).classed("faded", false);
    d3.selectAll([pc.canvas["brushed"]]).classed("full", false);
    d3.selectAll([pc.canvas["brushed"]]).classed("faded", true);
  } else {
    // updateProgressBar(active_data);
    $("#neighbors_list").empty();
  }
}
