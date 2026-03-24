// Menu Sidebar 
function openNav() {
    document.getElementById("mySidebar").style.width = "250px";
    document.body.style.backgroundColor = "rgba(0,0,0,0.4)";
}

function closeNav() {
    document.getElementById("mySidebar").style.width = "0";
    document.body.style.backgroundColor = "white";
}

function handleModal() {
  const closeBtn = document.getElementById('closeModalBtn')
  const background = document.getElementById('modalBackground')
  const content = document.getElementById('modalContent')
  const isClosed = sessionStorage.getItem('modalClosed');

  if (isClosed === 'true'){
    background.style.display = "none";
    content.style.display = "none";
  }

  if (closeBtn){
    closeBtn.addEventListener('click', () => {
      background.style.display = "none";
      content.style.display = "none";

      sessionStorage.setItem('modalClosed', 'true');
      console.log('Modal closed, and state saved permanently in localStorage.');
    })
  }
}

function instructionsBtn() {
  const btn = document.getElementById('openModalBtn')
  const background = document.getElementById('modalBackground')
  const content = document.getElementById('modalContent')

  btn.addEventListener('click', () => {
    background.style.display = "block";
    content.style.display = "block";
    
    sessionStorage.setItem('modalClosed', 'false');  
    console.log('Modal closed, and state saved permanently in localStorage.');
  })
}


// side panel function here
function openHist() {
  document.getElementById("sidePanelHistoryContent").style.width = "250px";
}

function closeHist() {
  document.getElementById("sidePanelHistoryContent").style.width = "0";
}




handleModal()
instructionsBtn()