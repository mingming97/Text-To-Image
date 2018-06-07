function $(id){
    return document.getElementById(id);
}

function timing(){
    var newtime = parseInt($('wait_time').innerHTML) - 1;
    $('wait_time').innerHTML = newtime;
}

window.onload=function(){
	var append = Math.random()*10;
    var clock = null;
    clock = window.setInterval('timing()',1000);

    window.setTimeout(function(){
        clearInterval(clock);
        cycle();
        $('wait_text1').innerHTML = '';
        $('wait_text2').innerHTML = '';
        $('wait_time').innerHTML = 'Finish!';
        $('img1').src = '../static/code/display/bird_0.png?data=' + append;
        $('img2').src = '../static/code/display/bird_0.png?data=' + append;
        $('img3').src = '../static/code/display/bird_1.png?data=' + append;
        $('img4').src = '../static/code/display/bird_2.png?data=' + append;
        $('img5').src = '../static/code/display/bird_3.png?data=' + append;
        $('img6').src = '../static/code/displaygen_pic/0.png?data=' + append;
        $('img7').src = '../static/code/displaygen_pic/1.png?data=' + append;
        $('img8').src = '../static/code/displaygen_pic/2.png?data=' + append;
        $('img9').src = '../static/code/displaygen_pic/3.png?data=' + append;
        $('img10').src = '../static/code/displaygen_pic/3.png?data=' + append;
    },50000);
}


function cycle(){
    var imgs_div=document.getElementById("imgs");
    var nav_div=document.getElementById("nav");
    //获取到图片轮播的ul对象数组
    var imgsUl=imgs_div.getElementsByTagName("ul")[0];
    //获取到远点的ul对象数组
    var nav=nav_div.getElementsByTagName("ul")[0];
    //上一个
    var prious=document.getElementById("preous");
    //下一个
    var next =document.getElementById("next");
    var timer;
    var animTimer;
    var index=1;
    play();
    prious.onclick=function(){
        initImgs(index);
        index-=1;
        if(index<1){
            index=8;
        }
        animate(256);
        btnShow(index);
    }
    next.onclick=function(){
        initImgs(index);
        index+=1;
        if(index>8){
            index=1;
        }
        animate(-256);
        btnShow(index);
    }

    function animate(offset){
        var newLeft=parseInt(imgsUl.offsetLeft)+offset;
        // imgsUl.style.left=newLeft;
        // console.log("定时器外面:此时offsetLeft"+imgsUl.offsetLeft+">>newLeft:"+newLeft);
        if(newLeft>-256){
            // imgsUl.style.left=-1024+"px";
            donghua(-2048);
        }else if(newLeft<-2048){
            // imgsUl.style.left=-256+"px";
            donghua(-256);
        }else{
            donghua(newLeft);
        }

    }
    function donghua(offset){
        clearInterval(animTimer);
        animTimer=setInterval(function(){
            imgsUl.style.left=imgsUl.offsetLeft+(offset-imgsUl.offsetLeft)/10 + "px";
            if(imgsUl.offsetLeft-offset<10&&imgsUl.offsetLeft-offset>-10){//如果偏移量已经等于指定好的偏移量，则清除定时器
                imgsUl.style.left=offset+"px";
                clearInterval(animTimer);
                //开启定时轮播
                play();
            }
        },20);
    }
    function initImgs(cur_index){
        clearInterval(timer);
        clearInterval(animTimer);
        var off=cur_index*256;
        imgsUl.style.left=-off+"px";
    }
    function play(){
        timer=setInterval(function(){
            next.onclick();
        },2000)
    }
    function btnShow(cur_index){
        var list=nav.children;
        for(var i=0;i<nav.children.length;i++){
            nav.children[i].children[0].className="hidden";
        }
        nav.children[cur_index-1].children[0].className="current";
    }
    for(var i=0;i<nav.children.length;i++){
        nav.children[i].index=i;
        var sd=nav.children[i].index;
        nav.children[i].onmouseover=function(){
            index=this.index+1;
            initImgs(this.index+1);
            btnShow(this.index+1);
        }
        nav.children[i].onmouseout=function(){
            play();
        }
    }
}
