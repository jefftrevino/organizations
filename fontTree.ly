date = #(strftime "%d-%m-%Y" (localtime (current-time)))
\header {
     title = \markup {
         \override #'(font-name . "Futura")
         "Organizations (For Cesar Chavez)"

     }
     
     opus = \markup {
         \override #'(font-name . "Futura")
         "Jeff Treviño, 2017"
     }
     tagline = \markup {
         \override #'(font-name . "Futura")
         \fontsize #-3.5
         {
             
             {  }
         } 
     }
}

\paper  {
  myStaffSize = #16
  #(define fonts
    (make-pango-font-tree "Futura"
                          "Futura"
                          "Futura"
                           (/ myStaffSize 16)))
}
