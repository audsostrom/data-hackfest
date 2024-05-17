'use client';
import {DM_Sans} from 'next/font/google';
import { createTheme } from '@mui/material/styles';

const dmSans = DM_Sans({
  weight: ['400', '500', '700'],
  subsets: ['latin'],
  display: 'swap',
});

const theme = createTheme({
  typography: {
    fontFamily: dmSans.style.fontFamily,
  },
  palette: {
    mode: 'dark',
    background: {
      default: '#141D26',
      paper: '#233446',
    },
  },
  components: {
    MuiButton: {
      variants: [
        {
          props: { variant: 'contained', color: 'primary' },
          style: {
            backgroundColor: "#08D354",
            color: 'black',
          },
        },
        {
          props: { variant: 'contained', color: 'secondary' },
          style: {
            backgroundColor: "#1E5EFF",
            color: 'white',
          },
        },
      ],
    },
  }
});

export default theme;
