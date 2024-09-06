particlesJS('particles-js', {
    particles: {
      number: {
        value: 80,
        density: {
          enable: true,
          value_area: 800
        }
      },
      color: {
        value: '#0ff'
      },
      shape: {
        type: 'circle',
        stroke: {
          width: 0,
          color: '#000000'
        }
      },
      opacity: {
        value: 0.5,
        random: true,
        anim: {
          enable: false,
          speed: 1,
          opacity_min: 0.1,
          sync: false
        }
      },
      size: {
        value: 3,
        random: true,
        anim: {
          enable: false,
          speed: 40,
          size_min: 0.1,
          sync: false
        }
      },
      line_linked: {
        enable: true,
        distance: 150,
        color: '#0ff',
        opacity: 0.4,
        width: 1
      },
      move: {
        enable: true,
        speed: 6,
        direction: 'none',
        random: false,
        straight: false,
        out_mode: 'out',
        bounce: false,
        attract: {
          enable: false,
          rotateX: 600,
          rotateY: 1200
        }
      }
    },
    interactivity: {
      detect_on: 'canvas',
      events: {
        onhover: {
          enable: true,
          mode: 'repulse'
        },
        onclick: {
          enable: true,
          mode: 'push'
        },
        resize: true
      },
      modes: {
        grab: {
          distance: 400,
          line_linked: {
            opacity: 1
          }
        },
        bubble: {
          distance: 400,
          size: 40,
          duration: 2,
          opacity: 8,
          speed: 3
        },
        repulse: {
          distance: 200,
          duration: 0.4
        },
        push: {
          particles_nb: 4
        },
        remove: {
          particles_nb: 2
        }
      }
    },
    retina_detect: true,
    z_index: 1
  });

let sliderCount = -1;  
// 初始化  
        addSlider();  
          function addSlider() {  
            sliderCount++;  

            // 创建一个新的输入框容器  
            var sliderContainer = document.createElement('div');  
            sliderContainer.className = 'slider-container';  

            // 创建标签  
            var titleLabel = document.createElement('label');  
            titleLabel.innerText = 'x^' + sliderCount;  

            // 创建当前值输入框  
            var valueInput = document.createElement('input');  
            valueInput.type = 'number';  
            valueInput.placeholder = '当前值';  
            valueInput.value = 1; // 默认值为1  

          

            // 添加元素到容器  
            sliderContainer.appendChild(titleLabel);  
            sliderContainer.appendChild(valueInput);  
              

            // 更新显示的多项式值  
            function updateSliderValuesDisplay() {  
                const sliderValues = Array.from(document.querySelectorAll('input[type="number"]')).map((input, index) => {  
                    const power = index; // 当前输入框的幂数  
                    return `${input.value}x^${power}`;   
                });  

                  
                document.getElementById('sliderValues').innerText = '当前f(x): ' + sliderValues.join(' + ');  
            }  

              
            valueInput.addEventListener('input', function() {    
                updateSliderValuesDisplay();   
            });              
            document.getElementById('sliders').appendChild(sliderContainer);  
            updateSliderValuesDisplay();  
        }  


         async function sendData() {  
            const sliderValues = Array.from(document.querySelectorAll('.output')).map((span, index) => {  
                return {power: index, value: parseFloat(span.innerText)};  
            });  

            await fetch('/solve', {  
                method: 'POST',  
                headers: {  
                    'Content-Type': 'application/json',  
                },  
                body: JSON.stringify(sliderValues),  
            })  
            .then(response => response.json())  
            .then(data => {  
                drawSolution(data);  
            })  
            .catch((error) => {  
                console.error('Error:', error);  
            });  
        }
        function drawSolution(data) {
            const xValues = data.x;
            const yValues = data.y;

            const xMin = Math.min(...xValues);
            const xMax = Math.max(...xValues);
            const yMin = Math.min(...yValues);
            const yMax = Math.max(...yValues);

            const xScale = 1 / (xMax - xMin);
            const yScale = 1 / (yMax - yMin);

            const plotlyData = [{
                type: 'scatter',
                mode: 'lines',
                x: xValues.map(x => (x - xMin) * xScale),
                y: yValues.map(y => 1 - ((y - yMin) * yScale)),
                line: {
                    color: '#007BFF',
                    width: 2
                }
            }];

            const plotlyLayout = {
                title: 'predicted function',
                xaxis: {
                    title: 'X ',
                },
                yaxis: {
                    title: 'u(x)',
                }
            };

            Plotly.newPlot('solutionCanvas', plotlyData, plotlyLayout);
                }