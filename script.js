document.addEventListener('DOMContentLoaded', () => {
  const navbarBurgers = document.querySelectorAll('.navbar-burger')
  const $navbarBurgers = Array.prototype.slice.call(navbarBurgers)

  if ($navbarBurgers.length > 0) {
    $navbarBurgers.forEach((el) => {
      el.addEventListener('click', () => {
        const target = el.dataset.target
        const $target = document.getElementById(target)

        el.classList.toggle('is-active')
        $target.classList.toggle('is-active')
      })
    })
  }
})
