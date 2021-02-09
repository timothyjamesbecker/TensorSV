var rest_host  = 'http://localhost:6080';
var sample_names = null,
    label_names = null,
    ftrs_idx = null,
    sample = null,
    label = null,
    max_num = null,
    tensors = null;

function render_tensor(id,data,dims){
    var margin = 10;
    var width = 5.0*document.getElementById('main_container').clientWidth/6.0;
    var height = 700;

    var svg = d3.select('#'+id).append('svg')
        .attr('width',width+margin)
        .attr('height',height+margin)
        .append('g')
        .attr('transform','translate('+margin+','+margin+')');
    var n_w = 11;
    var xy_data = [];
    var a = 0;
    for(var i=0;i<dims[0];i++){
        if (i>0 && i%n_w==0){ a = a+0;}
        for(var j=0;j<dims[1];j++){
            xy_data.push({'x':i+a,'y':j,'value':data[0][i][j]})
        }
    }

    var rw = width/dims[0],
        rh = height/dims[1];
    var xscale = d3.scaleLinear().range([0,width]).domain([0,dims[0]+5])
    var yscale = d3.scaleLinear().range([height-4*margin,0]).domain([0,dims[1]])

    function color_map(d){ //hsla(120,60%,70%,0.3) want to map -3 to 0 (R), 0 to 125 (G) and 3 to 250 (B)
        var clip = 1.0;
        var m1 = Math.max(-1*clip,Math.min(clip,d[0]));
        var m2 = Math.max(-1*clip,Math.min(clip,d[1]));
        var m3 = Math.max(-1*clip,Math.min(clip,d[2]));
        var m4 = Math.max(-1*clip,Math.min(clip,d[3]));
        if(m1<0.0){
            var color = 'hsl(0,'+(50-50*Math.abs(m2)/clip).toString()+'%,'+(100-50*Math.abs(m1)/clip).toString()+'%)';
        } else{
            var color = 'hsl(250,'+(100-50*Math.abs(m2)/clip).toString()+'%,'+(100-50*Math.abs(m1)/clip).toString()+'%)';
        }
        return color;
    }

    function curve_map(d,r){
        var clip = 1.0;
        var m2 = Math.abs(Math.max(-1*clip,Math.min(clip,d[1])));
        return r-r*(m2/1.0);
    }
    svg.selectAll('g')
        .data(xy_data) //select just first one for now
        .enter()
        .append('rect')
        .attr('stroke','black')
        .attr('stroke-width',0.5)
        .attr("x", function(d) { return xscale(d.x); })
        .attr("y", function(d) { return yscale(d.y); })
        .attr("rx", 0)//function(d){ return curve_map(d.value,rw); })
        .attr("ry", 0)//function(d){ return curve_map(d.value,rh); })
        .attr("width",rw)//function(d){ return rw/2+curve_map(d.value,rw); })
        .attr("height",rh)//function(d){ return rh/2+curve_map(d.value,rh); })
        .style("fill", function(d) { return color_map(d.value); } );
    // draw on column names

}

//data aquisition and processing functions
function get_tensors(url) {
    $.ajax({
        url: url,
        data: {},
        success: function (result) {
            if (result!=null) { //successfull REST response--------------------------------------------------------
                console.log(result);
                var data = result['data'][0];
                var dims = [];
                if(data.length>0){
                    if(data[0].length>0){
                        dims = [data.length,data[0].length,data[0][0].length];
                    }
                }
                render_tensor('tensor_container',result['data'],dims);
            }
        }
    });
}

//d3 drawing functions...
function render(e){
    if(e!=null){ e.preventDefault(); }
    var sm  = $('#tensor_form input[name=sms]').val();
    var lbl = $('#tensor_form input[name=lbls]').val();
    var num = $('#tensor_form input[name=num]').val();
    var url = rest_host+'/sample/'+sm+'/label/'+lbl+'/max_num/'+num
    get_tensors(url);
}

//build the DOM
$('#main_container').append('<div id="main_control_mask"></div>');
$('#main_control_mask').append('<div id="main_control_container"></div>');
$('#main_control_container').append('<form id="tensor_form">tensorSV\n' +
    '            <span class="ui-widget"><input id="sms_input" type="text" name="sms" class="sms_input"></span>\n' +
    '            <span class="ui-widget"><input id="lbls_input" type="text" name="lbls" class="sms_input"></span>\n' +
    '            <span class="ui-widget"><input id="num_input" type="text" name="num" class="num_input"></span>\n' +
    '            <input type="submit" id="view_button" class="tensor_input" value="view">\n' +
    '        </form>');
$('#tensor_form input[name=sms]').val('samples');
$('#tensor_form input[name=lbls]').val('labels');
$('#tensor_form input[name=num]').val('25');
$('#tensor_form').submit(function(e){
    render(e);
});

//maybe just use right and left arrow key?
// $('#main_control_container').append('<button id="prev_tensor">prev</button>');
// $('#main_control_container').append('<button id="next_tensor">next</button>');

$.ajax({
    url: rest_host+'/ftrs_idx',
    data:{},
    success:function(result){
        ftrs_idx = result;
    }
})

$.ajax({
    url: rest_host+'/sample_map',
    data: {},
    success: function (result) {
        $('#main_container').append('<div id="tensor_container"></div>');
        smpls = result;
        sample_names = Object.keys(smpls);
        lbl_names = {};
        console.log(sample_names);
        for(var i in sample_names){;
            var ls = smpls[sample_names[i]];
            for(var j in ls){ lbl_names[ls[j]] = 1; }
        }
        label_names = Object.keys(lbl_names);
        console.log(label_names);
        $( "#sms_input" )
            .on( "keydown", function( event ) {
                if ( event.keyCode === $.ui.keyCode.TAB &&
                    $( this ).autocomplete( "instance" ).menu.active ) {
                    event.preventDefault();
                }
            }).autocomplete({delay: 10, minLength: 0, source: sample_names});
        $( "#lbls_input" )
            .on( "keydown", function( event ) {
                if ( event.keyCode === $.ui.keyCode.TAB &&
                    $( this ).autocomplete( "instance" ).menu.active ) {
                    event.preventDefault();
                }
            }).autocomplete({delay: 10, minLength: 0, source: label_names});
    }
});