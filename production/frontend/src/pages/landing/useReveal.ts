import { useCallback } from 'react'

/** Ref callback: element gets class `revealed` the first time >=15% of it
 *  enters the viewport. Under reduced motion it reveals immediately. */
export function useReveal() {
  return useCallback((el: HTMLElement | null) => {
    if (!el) return
    if (
      typeof IntersectionObserver === 'undefined' ||
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ) {
      el.classList.add('revealed')
      return
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add('revealed')
            io.disconnect()
          }
        }
      },
      { threshold: 0.15 },
    )
    io.observe(el)
  }, [])
}
